/**
 * Optional `/model/infer.json` describes how Keras ordered class folders so the UI
 * maps the sigmoid to Dog vs Cat correctly (binary class 1 = alphabetically second folder).
 * It can also declare pixel normalization so inference matches Colab (`div255` vs MobileNet).
 */

/** Must match `write_infer_json` in `colab/pawdetect_colab_train.py`. */
export type PawdetectPixelNorm = "div255" | "mobilenet_v2";

export type PawdetectInferJson = {
  /** Folder names in Keras order: index 0 → label 0, index 1 → label 1 (sigmoid ≈ P(1)). */
  kerasClassIndexFoldersAlphabetical: [string, string];
  /**
   * `div255`: pixels ÷255 after resize (small CNN training).
   * `mobilenet_v2`: same as `tf.keras.applications.mobilenet_v2.preprocess_input` on 0–255 floats.
   */
  pixelNorm?: PawdetectPixelNorm;
};

const DEFAULT_INFER: PawdetectInferJson = {
  kerasClassIndexFoldersAlphabetical: ["cats", "dogs"],
  pixelNorm: "div255",
};

let activeInfer: PawdetectInferJson = DEFAULT_INFER;

function normalizePixelNorm(v: unknown): PawdetectPixelNorm {
  return v === "mobilenet_v2" ? "mobilenet_v2" : "div255";
}

export function setActiveInferConfig(config: PawdetectInferJson | null): void {
  if (!config || !Array.isArray(config.kerasClassIndexFoldersAlphabetical)) {
    activeInfer = DEFAULT_INFER;
    return;
  }
  const f = config.kerasClassIndexFoldersAlphabetical;
  if (f.length !== 2 || typeof f[0] !== "string" || typeof f[1] !== "string") {
    activeInfer = DEFAULT_INFER;
    return;
  }
  activeInfer = {
    kerasClassIndexFoldersAlphabetical: [f[0], f[1]],
    pixelNorm: normalizePixelNorm(config.pixelNorm),
  };
}

export function getActiveInferConfig(): PawdetectInferJson {
  return activeInfer;
}

export function getActivePixelNorm(): PawdetectPixelNorm {
  return normalizePixelNorm(activeInfer.pixelNorm);
}

/**
 * Colab MobileNet exports sometimes ship with `infer.json` still on `div255`.
 * If the loaded topology clearly contains MobileNet, force `mobilenet_v2` preprocessing.
 */
export function alignPixelNormWithModelTopology(
  modelTopology: object,
  infer: PawdetectInferJson | null,
): void {
  let looksMobile = false;
  try {
    looksMobile = JSON.stringify(modelTopology).toLowerCase().includes("mobilenet");
  } catch {
    return;
  }
  if (!looksMobile) return;
  if (normalizePixelNorm(infer?.pixelNorm) === "mobilenet_v2") return;

  if (process.env.NODE_ENV === "development" && typeof console !== "undefined") {
    console.warn(
      "[PawDetect] Model topology includes MobileNet but infer.json used div255 (or omitted pixelNorm). Using MobileNet v2 preprocessing (÷127.5 − 1). Update public/model/infer.json to pixelNorm \"mobilenet_v2\".",
    );
  }

  const folders: [string, string] =
    infer &&
    Array.isArray(infer.kerasClassIndexFoldersAlphabetical) &&
    infer.kerasClassIndexFoldersAlphabetical.length === 2 &&
    typeof infer.kerasClassIndexFoldersAlphabetical[0] === "string" &&
    typeof infer.kerasClassIndexFoldersAlphabetical[1] === "string"
      ? [infer.kerasClassIndexFoldersAlphabetical[0], infer.kerasClassIndexFoldersAlphabetical[1]]
      : DEFAULT_INFER.kerasClassIndexFoldersAlphabetical;

  setActiveInferConfig({ kerasClassIndexFoldersAlphabetical: folders, pixelNorm: "mobilenet_v2" });
}

/** Sigmoid from the model ≈ P(Keras class index 1). Convert to P(Dog) for the UI. */
export function rawSigmoidToDogProbability(raw: number): number {
  const [, class1] = activeInfer.kerasClassIndexFoldersAlphabetical;
  const n = class1.toLowerCase();
  if (n.includes("dog") && !n.includes("cat")) return raw;
  if (n.includes("cat") && !n.includes("dog")) return 1 - raw;
  return raw;
}

export function inferSummaryText(): string {
  const [c0, c1] = activeInfer.kerasClassIndexFoldersAlphabetical;
  const px = getActivePixelNorm();
  const pxLine =
    px === "mobilenet_v2"
      ? " Preprocess: MobileNet v2 (÷127.5 − 1 on 0–255 pixels)."
      : " Preprocess: pixels ÷255.";
  return `Training: “${c0}”→class 0, “${c1}”→class 1 (sigmoid ≈ P(class 1)).${pxLine}`;
}
