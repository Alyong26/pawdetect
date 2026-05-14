/**
 * OOD (out-of-distribution) pet detector built on top of MobileNet ImageNet.
 *
 * The binary cat-vs-dog head in `public/model/` was trained only on cats and
 * dogs. Even with a sigmoid margin threshold it can confidently misclassify
 * unrelated photos. To gate it reliably we run MobileNet (1k ImageNet classes)
 * and check the probability mass on the 118 dog classes (indices 151–268) and
 * the 5 cat classes (indices 281–285).
 *
 * Two MobileNet passes are performed for every photo:
 *   1. The full frame.
 *   2. A centered ~70% crop.
 *
 * The crop pass exists for partial-occlusion / multi-subject shots — selfies
 * with a pet, group photos, photos where the cat is centered but framed by
 * lots of background, etc. Whichever pass scores higher on the pet classes
 * is the one that drives the verdict.
 *
 * The final accept / reject decision happens in `predict.ts` so we can also
 * weigh in the binary head's confidence as a tiebreaker.
 */
import "@tensorflow/tfjs"; // ensure tfjs core is initialised before mobilenet
import type { MobileNet } from "@tensorflow-models/mobilenet";

/** ImageNet class index range for dog breeds (Chihuahua … Mexican hairless). */
const DOG_CLASS_MIN = 151;
const DOG_CLASS_MAX = 268;
/** ImageNet class indices for cat classes (tabby … Egyptian cat). */
const CAT_CLASS_INDICES = [281, 282, 283, 284, 285] as const;

export type PetDetectionKind = "dog" | "cat" | "neither";

export type PetDetectionResult = {
  /** Coarse verdict — "neither" iff both crop and full frame agree there is no pet. */
  kind: PetDetectionKind;
  /** Probability mass assigned to ImageNet dog classes (max over passes). */
  dogScore: number;
  /** Probability mass assigned to ImageNet cat classes (max over passes). */
  catScore: number;
  /** dogScore + catScore — primary "is this a pet" signal. */
  petScore: number;
  /** Best top-1 ImageNet label across passes. */
  topLabel: string;
  /** Probability of the top-1 label. */
  topProb: number;
};

let mobilenetPromise: Promise<MobileNet> | null = null;

/** Loads MobileNet v2 (alpha 0.5) once and caches the promise. */
export async function loadPetDetector(): Promise<MobileNet> {
  if (!mobilenetPromise) {
    mobilenetPromise = (async () => {
      const mobilenet = await import("@tensorflow-models/mobilenet");
      // version 2, alpha 0.5 is the lightest variant (~5 MB) — still very
      // accurate for the cat / dog gate we need.
      return mobilenet.load({ version: 2, alpha: 0.5 });
    })().catch((error) => {
      mobilenetPromise = null;
      throw error;
    });
  }
  return mobilenetPromise;
}

type PassScores = {
  dogScore: number;
  catScore: number;
  topLabel: string;
  topProb: number;
};

/**
 * Runs MobileNet on the full image AND on a centered crop, returning the
 * stronger signal. The orchestrator in `predict.ts` is responsible for the
 * final accept / reject decision.
 */
export async function detectPet(
  image: HTMLImageElement | ImageBitmap | HTMLCanvasElement | HTMLVideoElement,
): Promise<PetDetectionResult> {
  const model = await loadPetDetector();
  const renderable = toRenderable(image);

  const fullScores = await scoreImage(model, renderable);

  // Center crop pass: pets are often centered even when something else fills
  // the rest of the frame (a hand, hair, another person, background clutter).
  // Cropping to ~70% biases the classifier toward the central subject.
  const crop = centerCropToCanvas(renderable, 0.7);
  const cropScores = crop ? await scoreImage(model, crop) : null;

  const dogScore = Math.max(fullScores.dogScore, cropScores?.dogScore ?? 0);
  const catScore = Math.max(fullScores.catScore, cropScores?.catScore ?? 0);
  const petScore = dogScore + catScore;

  // Use whichever pass had the higher combined pet signal as the source of
  // the diagnostic top-label.
  const fullPet = fullScores.dogScore + fullScores.catScore;
  const cropPet = (cropScores?.dogScore ?? 0) + (cropScores?.catScore ?? 0);
  const top = cropPet > fullPet && cropScores ? cropScores : fullScores;

  const kind: PetDetectionKind =
    petScore <= 0 ? "neither" : dogScore >= catScore ? "dog" : "cat";

  return {
    kind,
    dogScore,
    catScore,
    petScore,
    topLabel: top.topLabel,
    topProb: top.topProb,
  };
}

async function scoreImage(
  model: MobileNet,
  renderable: HTMLImageElement | HTMLCanvasElement | HTMLVideoElement,
): Promise<PassScores> {
  const predictions = await model.classify(renderable, 5);
  let topLabel = "";
  let topProb = 0;
  for (const p of predictions) {
    if (p.probability > topProb) {
      topProb = p.probability;
      topLabel = p.className;
    }
  }

  const logits = model.infer(renderable, false) as import("@tensorflow/tfjs").Tensor;
  let dogScore = 0;
  let catScore = 0;
  try {
    const squeezed = logits.squeeze();
    const size = squeezed.size;
    // MobileNet v2 from tfjs-models emits [1, 1001] (background + 1k classes).
    // v1 emits [1, 1000]. Drop the background slot when present so our index
    // ranges line up with the canonical ImageNet ordering.
    const aligned = size === 1001 ? squeezed.slice([1], [1000]) : squeezed;
    const probs = (await aligned.data()) as Float32Array;

    let total = 0;
    for (let i = 0; i < probs.length; i += 1) total += probs[i];
    const isAlreadyProbs = Math.abs(total - 1) < 0.05;
    const probArray = isAlreadyProbs ? probs : softmax(probs);

    for (let i = DOG_CLASS_MIN; i <= DOG_CLASS_MAX; i += 1) {
      dogScore += probArray[i] ?? 0;
    }
    for (const i of CAT_CLASS_INDICES) {
      catScore += probArray[i] ?? 0;
    }

    if (aligned !== squeezed) aligned.dispose();
    squeezed.dispose();
  } finally {
    logits.dispose();
  }

  return { dogScore, catScore, topLabel, topProb };
}

function toRenderable(
  image: HTMLImageElement | ImageBitmap | HTMLCanvasElement | HTMLVideoElement,
): HTMLImageElement | HTMLCanvasElement | HTMLVideoElement {
  const isBitmap =
    typeof ImageBitmap !== "undefined" && image instanceof ImageBitmap;
  if (!isBitmap) {
    return image as HTMLImageElement | HTMLCanvasElement | HTMLVideoElement;
  }
  const bitmap = image as ImageBitmap;
  const canvas = document.createElement("canvas");
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const ctx = canvas.getContext("2d");
  if (ctx) ctx.drawImage(bitmap, 0, 0);
  return canvas;
}

function centerCropToCanvas(
  image: HTMLImageElement | HTMLCanvasElement | HTMLVideoElement,
  ratio: number,
): HTMLCanvasElement | null {
  const { width, height } = naturalSize(image);
  if (width < 32 || height < 32) return null;
  const minDim = Math.min(width, height);
  const cropSize = Math.max(16, Math.floor(minDim * ratio));
  const sx = Math.floor((width - cropSize) / 2);
  const sy = Math.floor((height - cropSize) / 2);
  const canvas = document.createElement("canvas");
  canvas.width = cropSize;
  canvas.height = cropSize;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.drawImage(image, sx, sy, cropSize, cropSize, 0, 0, cropSize, cropSize);
  return canvas;
}

function naturalSize(
  image: HTMLImageElement | HTMLCanvasElement | HTMLVideoElement,
): { width: number; height: number } {
  if (image instanceof HTMLImageElement) {
    return { width: image.naturalWidth, height: image.naturalHeight };
  }
  if (image instanceof HTMLVideoElement) {
    return { width: image.videoWidth, height: image.videoHeight };
  }
  return { width: image.width, height: image.height };
}

function softmax(values: Float32Array): Float32Array {
  let max = -Infinity;
  for (let i = 0; i < values.length; i += 1) {
    if (values[i] > max) max = values[i];
  }
  const out = new Float32Array(values.length);
  let sum = 0;
  for (let i = 0; i < values.length; i += 1) {
    out[i] = Math.exp(values[i] - max);
    sum += out[i];
  }
  for (let i = 0; i < values.length; i += 1) {
    out[i] /= sum;
  }
  return out;
}
