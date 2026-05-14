import { writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { loadTf } from "./load-tfjs.mjs";
import { buildModel, saveLayersModel } from "./tfjs-cnn-shared.mjs";

/**
 * Writes a valid (randomly initialised) CNN to `public/model`.
 * Run `npm run train:model` afterwards if you need meaningful accuracy.
 */

const tf = await loadTf();

const exportDir = path.join(process.cwd(), "public", "model");
const model = buildModel(tf);

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

console.log(`Wrote fresh TF.js artifacts to ${exportDir}`);
