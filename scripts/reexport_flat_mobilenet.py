#!/usr/bin/env python3
"""Rebuild `public/model` with a flat MobileNetV2 graph (see `build_mobilenet_model` in Colab).

TensorFlow.js 4.x fails on nested Functional Keras 3 exports (`Graph disconnected` at the inner
`InputLayer`). Training uses the same flat architecture as Colab after the Colab script update.

Requires: `tensorflow`, `tensorflowjs` (see repo venv or Colab).

Usage (from repo root):

  .venv_spyder\\Scripts\\python scripts\\reexport_flat_mobilenet.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _functional_count(obj: object) -> int:
    n = 0
    if isinstance(obj, dict):
        if obj.get("class_name") == "Functional":
            n += 1
        for v in obj.values():
            n += _functional_count(v)
    elif isinstance(obj, list):
        for x in obj:
            n += _functional_count(x)
    return n


def main() -> int:
    sys.path.insert(0, str(REPO / "colab"))
    from pawdetect_colab_train import (  # noqa: E402
        build_mobilenet_model,
        export_tfjs,
        finalize_tfjs_export,
    )

    out = REPO / "public" / "model"
    infer_backup = (out / "infer.json").read_text(encoding="utf-8")
    staging = REPO / "public" / ".model_export_tmp"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    model = build_mobilenet_model()
    cfg = json.loads(model.to_json())
    fc = _functional_count(cfg)
    if fc != 1:
        raise SystemExit(
            f"Expected exactly one Functional in model JSON for tfjs compatibility; found {fc}."
        )

    export_tfjs(model, staging)
    finalize_tfjs_export(staging)

    (staging / "infer.json").write_text(infer_backup, encoding="utf-8")

    for name in ("model.json", "group1-shard.bin", "infer.json"):
        src = staging / name
        if not src.is_file():
            raise SystemExit(f"Missing export artifact: {src}")
        shutil.move(src, out / name)

    shutil.rmtree(staging)
    print("Wrote flat MobileNet export to", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
