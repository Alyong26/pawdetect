import { readdir, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import sharp from "sharp";
import { loadTf } from "./load-tfjs.mjs";
import { buildModel, saveLayersModel } from "./tfjs-cnn-shared.mjs";

/**
 * Trains the PawDetect CNN on the Kaggle-style folder layout using `model.fitDataset`
 * and an async batch generator (never loads the full dataset into one tensor).
 *
 * Expected layout:
 *   <repo>/_dataset_extract/training_set/training_set/cats/*.jpg
 *   <repo>/_dataset_extract/training_set/training_set/dogs/*.jpg
 *
 * CPU training is still slow on large archives; for faster turnaround use the Colab
 * notebook in `colab/PawDetect_train.ipynb` (GPU) and copy the export into `public/model`.
 */

const tf = await loadTf();

const root = process.cwd();
const catsDir = path.join(root, "_dataset_extract", "training_set", "training_set", "cats");
const dogsDir = path.join(root, "_dataset_extract", "training_set", "training_set", "dogs");
const exportDir = path.join(root, "public", "model");

const IMAGE_EXTS = new Set([".jpg", ".jpeg", ".png", ".webp", ".gif"]);

function shuffleInPlace(array) {
  for (let i = array.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
}

async function listImages(dir) {
  const entries = await readdir(dir, { withFileTypes: true });
  return entries
    .filter((entry) => entry.isFile() && IMAGE_EXTS.has(path.extname(entry.name).toLowerCase()))
    .map((entry) => path.join(dir, entry.name));
}

async function fileToTensor(filePath) {
  const { data, info } = await sharp(filePath)
    .resize(224, 224, { fit: "fill" })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const channels = info.channels;
  const floats = new Float32Array(info.width * info.height * 3);
  let dst = 0;
  for (let src = 0; src < data.length; src += channels) {
    floats[dst++] = data[src] / 255;
    floats[dst++] = data[src + 1] / 255;
    floats[dst++] = data[src + 2] / 255;
  }

  return tf.tensor3d(floats, [224, 224, 3]).expandDims(0);
}

async function buildBatchTensor(samples) {
  const tensors = [];
  for (const s of samples) {
    tensors.push(await fileToTensor(s.file));
  }
  const xs = tf.concat(tensors, 0);
  tensors.forEach((t) => t.dispose());
  const ys = tf.tensor2d(
    samples.map((s) => [s.label]),
    [samples.length, 1],
  );
  return { xs, ys };
}

async function loadBalancedFileList(maxPerClass) {
  const catFiles = await listImages(catsDir);
  const dogFiles = await listImages(dogsDir);

  if (!catFiles.length || !dogFiles.length) {
    throw new Error(
      `Could not find training images. Expected cats in ${catsDir} and dogs in ${dogsDir}.`,
    );
  }

  shuffleInPlace(catFiles);
  shuffleInPlace(dogFiles);

  const cap =
    maxPerClass > 0 ? maxPerClass : Math.min(catFiles.length, dogFiles.length);

  const cats = catFiles.slice(0, Math.min(cap, catFiles.length));
  const dogs = dogFiles.slice(0, Math.min(cap, dogFiles.length));

  const files = [
    ...cats.map((file) => ({ file, label: 0 })),
    ...dogs.map((file) => ({ file, label: 1 })),
  ];

  shuffleInPlace(files);
  return { files, catsUsed: cats.length, dogsUsed: dogs.length };
}

const MAX_PER_CLASS = Number(process.env.PAWDETECT_MAX_PER_CLASS ?? "0");
const EPOCHS = Number(process.env.PAWDETECT_EPOCHS ?? 8);
const BATCH = Number(process.env.PAWDETECT_BATCH ?? 12);

const { files, catsUsed, dogsUsed } = await loadBalancedFileList(MAX_PER_CLASS);
console.log(
  `Training on ${files.length} images (cats=${catsUsed}, dogs=${dogsUsed}), batch=${BATCH}, epochs=${EPOCHS}`,
);

const model = buildModel(tf);

const dataset = tf.data.generator(async function* () {
  shuffleInPlace(files);
  for (let i = 0; i < files.length; i += BATCH) {
    const slice = files.slice(i, i + BATCH);
    const { xs, ys } = await buildBatchTensor(slice);
    yield { xs, ys };
  }
});

await model.fitDataset(dataset, {
  epochs: EPOCHS,
  callbacks: {
    onEpochEnd: async (epoch, logs) => {
      console.log(
        `Epoch ${epoch + 1}/${EPOCHS} — loss: ${logs?.loss?.toFixed(4) ?? "?"} acc: ${logs?.acc?.toFixed(4) ?? "?"}`,
      );
    },
  },
});

await saveLayersModel(tf, model, exportDir);
await writeFile(
  path.join(exportDir, "infer.json"),
  `${JSON.stringify(
    {
      kerasClassIndexFoldersAlphabetical: ["cats", "dogs"],
      pixelNorm: "div255",
    },
    null,
    2,
  )}\n`,
);
model.dispose();

console.log(`Exported TF.js model to ${exportDir}`);
