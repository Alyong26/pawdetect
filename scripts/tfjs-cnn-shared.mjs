import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

/**
 * Shared CNN definition + disk export used by both `export-initial-model.mjs`
 * (fast bootstrap) and `train-export-tfjs.mjs` (full training).
 *
 * Architecture matches `colab/pawdetect_colab_train.py` (BN + L2 head) so Colab exports
 * and local tooling stay compatible.
 *
 * @param {import("@tensorflow/tfjs")} tf — from `loadTf()` (native or pure JS).
 */

function convBnPool(tf, filters) {
  return [
    tf.layers.conv2d({
      filters,
      kernelSize: 3,
      padding: "same",
      useBias: false,
    }),
    tf.layers.batchNormalization(),
    tf.layers.reLU(),
    tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }),
  ];
}

export function buildModel(tf) {
  const l2 = tf.regularizers.l2({ l2: 1e-4 });
  const model = tf.sequential({
    layers: [
      tf.layers.inputLayer({ inputShape: [224, 224, 3] }),
      ...convBnPool(tf, 32),
      ...convBnPool(tf, 64),
      ...convBnPool(tf, 128),
      tf.layers.globalAveragePooling2d({}),
      tf.layers.dropout({ rate: 0.35 }),
      tf.layers.dense({
        units: 1,
        activation: "sigmoid",
        kernelRegularizer: l2,
      }),
    ],
  });

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
}

export async function saveLayersModel(tf, model, targetDir) {
  const handler = tf.io.withSaveHandler(async (artifacts) => {
    if (!artifacts.modelTopology) {
      throw new Error("Missing modelTopology — cannot export this model type.");
    }
    if (!artifacts.weightSpecs || !artifacts.weightData) {
      throw new Error("Missing weights — did the model finish training?");
    }

    const buffers = Array.isArray(artifacts.weightData)
      ? artifacts.weightData.map((item) => Buffer.from(item))
      : [Buffer.from(artifacts.weightData)];

    const shardName = "group1-shard.bin";
    const weightsManifest = [
      {
        paths: [shardName],
        weights: artifacts.weightSpecs,
      },
    ];

    const manifest = {
      format: "layers-model",
      generatedBy: `TensorFlow.js tfjs-core ${tf.version_core}`,
      convertedBy: "PawDetect tooling",
      modelTopology: artifacts.modelTopology,
      weightsManifest,
      trainingConfig: artifacts.trainingConfig,
      userDefinedMetadata: artifacts.userDefinedMetadata,
    };

    await mkdir(targetDir, { recursive: true });
    await writeFile(path.join(targetDir, "model.json"), JSON.stringify(manifest, null, 2));
    await writeFile(path.join(targetDir, shardName), Buffer.concat(buffers));
    return { modelArtifactsInfo: { dateSaved: new Date().toISOString(), modelTopologyType: "JSON" } };
  });

  await model.save(handler);
}
