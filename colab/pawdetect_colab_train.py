#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PawDetect — train on Google Colab (GPU) and export TensorFlow.js layers model.

Usage on Colab (example):
  1. Runtime → Change runtime type → GPU (T4 is enough).
  2. Upload your `archive.zip` (cats/dogs folder layout).
  3. Install deps (pick one approach):

     **A — Minimal (try first).** Colab already includes TensorFlow; often you only need:

     !pip install -q tensorflowjs

     **B — Pin TF + tfjs** if you need a specific range:

     !pip install -q "tensorflow>=2.16,<2.20" "tensorflowjs>=4.11,<5"

     **C — Quiet Colab’s TF 2.20 stack warnings (optional).** After `tensorflowjs`, pip may
     report conflicts with `tensorflow-text`, `ydf-tf`, `packaging`, etc. PawDetect only
     needs `tensorflow` + `tensorflowjs`. If imports work, you can ignore the noise. To
     align versions and reduce warnings:

     !pip install -q "tensorflow==2.20.0" "tensorflowjs>=4.11,<5" "packaging>=24.2"

     Pip may print **red “dependency conflict” lines** about Colab’s *other* preinstalled
     packages (`tensorflow-text`, `google-cloud-*`, etc.). That is normal: the cell can
     still succeed (green check). Verify in the next cell:

     import tensorflow as tf
     import tensorflowjs as tfjs
     print("TF", tf.__version__, "| tfjs", tfjs.__version__)

     If imports fail: **Runtime → Restart session**, run **only** the pip cell first, then continue.

  4. Upload and unzip your dataset, then run this script:

     from google.colab import files
     files.upload()  # pick archive.zip

     !mkdir -p /content/pawdetect && unzip -oq archive.zip -d /content/pawdetect

     # If your zip matches this repo, paths are often:
     #   /content/pawdetect/training_set/training_set/cats
     #   /content/pawdetect/training_set/training_set/dogs

     !PAWDETECT_DATA=/content/pawdetect/training_set/training_set python pawdetect_colab_train.py

     Default **PAWDETECT_ARCH=mobilenet** uses ImageNet MobileNetV2 + `infer.json` `pixelNorm`:
     `"mobilenet_v2"` so the Next.js app matches preprocessing. Use `PAWDETECT_ARCH=scratch` for
     the small CNN (pixels÷255 only).

  5. Download `pawdetect_tfjs.zip` and unzip into your repo's `public/model/` (overwrite
     `model.json`, **`group1-shard.bin`** — the script merges any multi-shard export into one
     file — and `infer.json`).

Environment:
  PAWDETECT_DATA   root folder that contains `cats/` and `dogs/` subfolders (default below).
  PAWDETECT_ARCH   `mobilenet` (default) or `scratch` for the small CNN from earlier versions.
  PAWDETECT_EPOCHS (default 40; try 60+ if early stopping allows)
  PAWDETECT_BATCH  (default 64; reduce to 32 on OOM)
  PAWDETECT_LR     (default 0.0001 for mobilenet, 0.001 for scratch; override as needed)
  PAWDETECT_VAL_SPLIT  fraction held out for validation (default 0.15)
"""

from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path

# Default matches the Kaggle-style archive used in PawDetect.
DEFAULT_DATA = "/content/pawdetect/training_set/training_set"

EPOCHS = int(os.environ.get("PAWDETECT_EPOCHS", "40"))
BATCH = int(os.environ.get("PAWDETECT_BATCH", "64"))
VAL_SPLIT = float(os.environ.get("PAWDETECT_VAL_SPLIT", "0.15"))

ARCH = os.environ.get("PAWDETECT_ARCH", "mobilenet").strip().lower()
USE_SCRATCH = ARCH in ("scratch", "cnn", "small")
_DEFAULT_LR = 0.001 if USE_SCRATCH else 0.0001
LR = float(os.environ.get("PAWDETECT_LR", str(_DEFAULT_LR)))


def build_scratch_model():
    """Small CNN trained on pixels÷255 (matches legacy `pixelNorm: div255`)."""
    import tensorflow as tf

    def conv_bn_pool(filters: int) -> list:
        return [
            tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.MaxPooling2D(2),
        ]

    blocks: list = [tf.keras.layers.Input(shape=(224, 224, 3))]
    for f in (32, 64, 128):
        blocks.extend(conv_bn_pool(f))
    blocks.extend(
        [
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(
                1,
                activation="sigmoid",
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            ),
        ]
    )

    model = tf.keras.Sequential(blocks, name="pawdetect_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def build_mobilenet_model():
    """MobileNetV2 transfer learning; train with `mobilenet_v2.preprocess_input` (see make_datasets).

    Uses ``input_tensor=inp`` so MobileNet is inlined into one Functional graph. Wrapping
    ``base(inp)`` produces a *nested* Functional in ``model.json`` that TensorFlow.js 4.x
    fails to deserialize (graph disconnect at the inner ``InputLayer``). The flat graph
    loads correctly in the browser with the same Keras3/tfjs JSON patches in ``loadModel.ts``.
    """
    import tensorflow as tf

    inp = tf.keras.Input(shape=(224, 224, 3))
    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
        input_tensor=inp,
    )
    base.trainable = False
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dropout(0.25)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inp, out, name="pawdetect_mobilenetv2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def make_datasets(data_dir: Path, *, preprocess_mode: str):
    import tensorflow as tf

    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Missing dataset folder: {data_dir}\n"
            f"Set PAWDETECT_DATA to the directory that contains cats/ and dogs/."
        )

    if preprocess_mode not in ("div255", "mobilenet_v2"):
        raise ValueError(f"Unknown preprocess_mode: {preprocess_mode!r}")

    seed = 42
    kwargs = dict(
        directory=str(data_dir),
        labels="inferred",
        label_mode="binary",
        image_size=(224, 224),
        interpolation="bilinear",
        batch_size=BATCH,
        seed=seed,
        validation_split=VAL_SPLIT,
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(subset="training", shuffle=True, **kwargs)
    val_ds = tf.keras.utils.image_dataset_from_directory(subset="validation", shuffle=False, **kwargs)

    def augment_train_images(x, y):
        """Match eval pipeline after: flip + mild photometric jitter (still 0–255 before /255)."""
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_brightness(x, max_delta=0.12)
        x = tf.image.random_contrast(x, lower=0.88, upper=1.12)
        x = tf.clip_by_value(x, 0.0, 255.0)
        return x, y

    train_ds = train_ds.map(augment_train_images, num_parallel_calls=tf.data.AUTOTUNE)
    if preprocess_mode == "mobilenet_v2":
        def pre_mobilenet(x, y):
            return tf.keras.applications.mobilenet_v2.preprocess_input(x), y

        train_ds = train_ds.map(pre_mobilenet, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(pre_mobilenet, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        norm = tf.keras.layers.Rescaling(1.0 / 255.0)
        train_ds = train_ds.map(lambda x, y: (norm(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(lambda x, y: (norm(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds


def export_tfjs(model, export_dir: Path) -> None:
    try:
        import tensorflowjs as tfjs
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Missing package: tensorflowjs. In Colab run this in a cell *before* this script:\n"
            "  !pip install -q tensorflowjs\n"
            "If imports still fail: Runtime → Restart session, run only the pip cell, then this script."
        ) from e

    export_dir.mkdir(parents=True, exist_ok=True)
    # Default tfjs shard size is ~4MB; MobileNet can split into several files. Prefer one shard
    # at export time; `finalize_tfjs_export` still merges if multiple paths remain.
    try:
        tfjs.converters.save_keras_model(
            model,
            str(export_dir),
            weight_shard_size_bytes=512 * 1024 * 1024,
        )
    except TypeError:
        tfjs.converters.save_keras_model(model, str(export_dir))


def _patch_sequential_weight_manifest_names(data: dict) -> None:
    """Strip Sequential container prefix from manifest names (Keras 3 vs tfjs-layers)."""
    try:
        seq_name = data["modelTopology"]["model_config"]["config"]["name"]
    except (KeyError, TypeError):
        return
    if not isinstance(seq_name, str) or not seq_name:
        return
    prefix = f"{seq_name}/"
    for group in data.get("weightsManifest") or []:
        if not isinstance(group, dict):
            continue
        for w in group.get("weights") or []:
            if not isinstance(w, dict):
                continue
            n = w.get("name")
            if isinstance(n, str) and n.startswith(prefix):
                w["name"] = n[len(prefix) :]


_REG_KEYS_FOR_TFJS = ("kernel_regularizer", "bias_regularizer", "activity_regularizer")


def _strip_keras_regularizers_for_tfjs(obj: object) -> None:
    """tfjs-layers cannot deserialize Keras L1/L2 regularizer blocks; they do not affect inference."""
    if isinstance(obj, dict):
        for k in _REG_KEYS_FOR_TFJS:
            obj.pop(k, None)
        for v in obj.values():
            _strip_keras_regularizers_for_tfjs(v)
    elif isinstance(obj, list):
        for x in obj:
            _strip_keras_regularizers_for_tfjs(x)


def _patch_keras3_input_layer_batch_shape(obj: object) -> None:
    """Keras 3 / tfjs converter writes InputLayer `batch_shape`; TF.js expects `batchInputShape`."""
    if isinstance(obj, dict):
        if obj.get("class_name") == "InputLayer" and isinstance(obj.get("config"), dict):
            cfg = obj["config"]
            if "batch_shape" in cfg and "batchInputShape" not in cfg:
                cfg["batchInputShape"] = cfg.pop("batch_shape")
        for v in obj.values():
            _patch_keras3_input_layer_batch_shape(v)
    elif isinstance(obj, list):
        for x in obj:
            _patch_keras3_input_layer_batch_shape(x)


def _keras3_collect_tensor_specs(args: object) -> list:
    """Flatten nested `args` (e.g. Add layer `[[tensorA, tensorB]]`) into __keras_tensor__ dicts."""
    specs: list = []

    def visit(node: object) -> None:
        if isinstance(node, list):
            for x in node:
                visit(x)
            return
        if not isinstance(node, dict):
            return
        cn = node.get("class_name") or node.get("className")
        if cn in ("__keras_tensor__", "KerasTensor"):
            specs.append(node)

    visit(args)
    return specs


def _keras3_inbound_node_to_legacy(node: object) -> object:
    """Keras 3 node {args, kwargs} -> tfjs list of [layer, nodeIdx, tensorIdx, optionalKwargs]."""
    if isinstance(node, list):
        return node
    if not isinstance(node, dict) or "args" not in node:
        return node
    tuples: list = []
    kwargs_all = node.get("kwargs") or {}
    has_kw = isinstance(kwargs_all, dict) and len(kwargs_all) > 0
    for arg in _keras3_collect_tensor_specs(node.get("args")):
        if not isinstance(arg, dict):
            continue
        cn = arg.get("class_name") or arg.get("className")
        if cn not in ("__keras_tensor__", "KerasTensor"):
            continue
        cfg = arg.get("config")
        if not isinstance(cfg, dict):
            continue
        hist = cfg.get("keras_history") or cfg.get("kerasHistory")
        if isinstance(hist, dict):
            h = hist
            layer_name = h.get("layer") or h.get("0")
            n_i = h.get("node_index", h.get("nodeIndex", h.get("1")))
            t_i = h.get("tensor_index", h.get("tensorIndex", h.get("2")))
            if layer_name is None or n_i is None or t_i is None:
                continue
            hist = (layer_name, n_i, t_i)
        if not isinstance(hist, (list, tuple)) or len(hist) < 3:
            continue
        layer_name, n_i, t_i = hist[0], hist[1], hist[2]
        if not isinstance(layer_name, str):
            continue
        row: list = [layer_name, int(n_i), int(t_i)]
        if has_kw:
            row.append(kwargs_all)
        tuples.append(row)
    if not tuples:
        return node
    return tuples


def _record_numeric_keys_to_list(v: dict) -> list | None:
    keys = list(v.keys())
    if not keys:
        return None
    if not all(isinstance(k, str) and k.isdigit() for k in keys):
        return None
    keys.sort(key=lambda x: int(x))
    return [v[k] for k in keys]


def _patch_keras3_inbound_nodes_for_tfjs(obj: object) -> None:
    """tfjs-layers expects inboundNodes entries to be arrays, not Keras 3 {args, kwargs} dicts."""
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            if k in ("inbound_nodes", "inboundNodes"):
                if isinstance(v, list):
                    obj[k] = [_keras3_inbound_node_to_legacy(n) for n in v]
                elif isinstance(v, dict):
                    as_arr = _record_numeric_keys_to_list(v)
                    if as_arr is not None:
                        obj[k] = [_keras3_inbound_node_to_legacy(n) for n in as_arr]
                    elif "args" in v:
                        obj[k] = [_keras3_inbound_node_to_legacy(v)]
                    else:
                        _patch_keras3_inbound_nodes_for_tfjs(v)
            else:
                _patch_keras3_inbound_nodes_for_tfjs(v)
    elif isinstance(obj, list):
        for x in obj:
            _patch_keras3_inbound_nodes_for_tfjs(x)


def _patch_keras3_functional_io_lists(obj: object) -> None:
    """Keras 3 may emit flat [name, node, tensor]; tfjs-layers expects [[name, node, tensor]]."""
    keys = ("input_layers", "inputLayers", "output_layers", "outputLayers")
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            if k in keys:
                v = obj[k]
                if (
                    isinstance(v, list)
                    and len(v) == 3
                    and isinstance(v[0], str)
                    and isinstance(v[1], (int, float))
                    and isinstance(v[2], (int, float))
                ):
                    obj[k] = [[v[0], int(v[1]), int(v[2])]]
            else:
                _patch_keras3_functional_io_lists(obj[k])
    elif isinstance(obj, list):
        for x in obj:
            _patch_keras3_functional_io_lists(x)


def finalize_tfjs_export(export_dir: Path) -> None:
    """Patch model.json for tfjs-layers; merge every listed weight shard into one file per group."""
    model_json = export_dir / "model.json"
    data = json.loads(model_json.read_text(encoding="utf-8"))
    _patch_keras3_input_layer_batch_shape(data)
    _strip_keras_regularizers_for_tfjs(data)
    _patch_keras3_inbound_nodes_for_tfjs(data)
    _patch_keras3_functional_io_lists(data)
    _patch_sequential_weight_manifest_names(data)

    manifest = data.get("weightsManifest")
    if not isinstance(manifest, list) or len(manifest) == 0:
        model_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return

    single_group = len(manifest) == 1
    for idx, group in enumerate(manifest):
        if not isinstance(group, dict):
            continue
        paths = group.get("paths")
        if not isinstance(paths, list) or len(paths) == 0:
            continue
        out_name = "group1-shard.bin" if single_group else f"group{idx + 1}-shard.bin"
        blobs: list[bytes] = []
        for name in paths:
            if not isinstance(name, str):
                continue
            shard = export_dir / name
            if not shard.is_file():
                raise FileNotFoundError(
                    f"Missing weight shard listed in model.json: {shard.name} (expected {shard})"
                )
            blobs.append(shard.read_bytes())
        merged = b"".join(blobs)
        out_path = export_dir / out_name

        for name in paths:
            if not isinstance(name, str):
                continue
            old = export_dir / name
            if old.is_file() and old.resolve() != out_path.resolve():
                old.unlink()

        out_path.write_bytes(merged)
        group["paths"] = [out_name]
        print(
            f"TensorFlow.js weights: merged {len(blobs)} shard(s) -> {out_name} ({len(merged)} bytes)"
        )

    model_json.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_infer_json(export_dir: Path, data_root: Path, pixel_norm: str) -> None:
    """Browser maps sigmoid using Keras alphabetical order; pixel_norm must match preprocessing."""
    classes = sorted(p.name for p in data_root.iterdir() if p.is_dir())
    if len(classes) < 2:
        raise ValueError(f"Need ≥2 class subfolders under {data_root}; found: {classes!r}")
    infer = {
        "kerasClassIndexFoldersAlphabetical": [classes[0], classes[1]],
        "pixelNorm": pixel_norm,
    }
    (export_dir / "infer.json").write_text(json.dumps(infer, indent=2) + "\n", encoding="utf-8")
    print("infer.json (for Next.js):", infer)


def zip_export(export_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in export_dir.iterdir():
            if f.is_file():
                zf.write(f, arcname=f.name)


def main() -> int:
    import tensorflow as tf

    data_root = Path(os.environ.get("PAWDETECT_DATA", DEFAULT_DATA)).resolve()
    export_dir = Path("/content/pawdetect_tfjs_export")
    zip_path = Path("/content/pawdetect_tfjs.zip")

    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True)

    print("TensorFlow:", tf.__version__)
    print("Data dir:", data_root)
    preprocess_mode = "div255" if USE_SCRATCH else "mobilenet_v2"
    print("PAWDETECT_ARCH:", ARCH, "| preprocess:", preprocess_mode, "| LR:", LR)
    class_dirs = sorted(p.name for p in data_root.iterdir() if p.is_dir())
    print("Class folders (alphabetical → Keras binary 0, then 1):", class_dirs)
    if len(class_dirs) >= 2:
        print(f"  sigmoid ≈ P('{class_dirs[1]}')  [label 1]; 1−sigmoid ≈ P('{class_dirs[0]}')")
    print(f"Epochs={EPOCHS} batch={BATCH} val_split={VAL_SPLIT}")

    train_ds, val_ds = make_datasets(data_root, preprocess_mode=preprocess_mode)
    model = build_scratch_model() if USE_SCRATCH else build_mobilenet_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            min_delta=1e-4,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

    export_tfjs(model, export_dir)
    finalize_tfjs_export(export_dir)
    write_infer_json(export_dir, data_root, preprocess_mode)
    zip_export(export_dir, zip_path)

    print("Wrote:", zip_path)
    print("Download it, unzip into your project's public/model/ on Windows/macOS/Linux.")
    try:
        from google.colab import files as colab_files

        colab_files.download(str(zip_path))
    except ImportError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
