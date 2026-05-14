/**
 * OOD (out-of-distribution) pet detector built on top of MobileNet ImageNet.
 *
 * The binary cat-vs-dog head we ship in `public/model/` has never seen images
 * that are neither cats nor dogs. Even with a probability margin threshold it
 * can return Dog or Cat with very high confidence on an unrelated photo
 * (a car, a flower, a landscape, …). Thresholding sigmoid is not enough.
 *
 * To gate the binary head reliably, we run MobileNet (ImageNet, 1k classes)
 * first and check whether any dog-breed class (indices 151–268) or cat class
 * (indices 281–285) carries non-trivial mass. If neither does, the photo is
 * almost certainly not a cat or dog and we short-circuit to "Not a Dog or Cat".
 *
 * MobileNet v2 with alpha 0.5 keeps the model footprint small (~5 MB) and is
 * lazy-loaded only the first time a user clicks Classify, so the initial page
 * load stays snappy.
 */
import "@tensorflow/tfjs"; // ensure tfjs core is initialised before mobilenet
import type { MobileNet } from "@tensorflow-models/mobilenet";

/** ImageNet class index range for dog breeds (Chihuahua … Mexican hairless). */
const DOG_CLASS_MIN = 151;
const DOG_CLASS_MAX = 268;
/** ImageNet class indices for cat classes (tabby … Egyptian cat). */
const CAT_CLASS_INDICES = [281, 282, 283, 284, 285] as const;

/**
 * Minimum combined dog+cat probability needed to even consider running the
 * binary head. Below this the prediction is reported as "Not a Dog or Cat".
 * Empirically:
 *   - Clear pet photos: petScore is usually 0.4–0.95.
 *   - Random non-pet photos (cars, landscapes, people, food …): petScore < 0.05.
 *   - Borderline animals (wolves, foxes, tigers): petScore is in the 0.05–0.15 range.
 *
 * 0.12 keeps the gate strict without rejecting unusual angles or
 * indoor lighting of real pets.
 */
const PET_SCORE_THRESHOLD = 0.12;

export type PetDetectionKind = "dog" | "cat" | "neither";

export type PetDetectionResult = {
  kind: PetDetectionKind;
  /** Probability mass assigned to ImageNet dog classes. */
  dogScore: number;
  /** Probability mass assigned to ImageNet cat classes. */
  catScore: number;
  /** dogScore + catScore. */
  petScore: number;
  /** Best top-1 ImageNet label, for diagnostics. */
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
      // Reset so the next call re-attempts; otherwise we cache the failure.
      mobilenetPromise = null;
      throw error;
    });
  }
  return mobilenetPromise;
}

/**
 * Runs MobileNet on the supplied image and returns the dog/cat probability
 * masses plus a verdict. The image is classified at native resolution.
 */
export async function detectPet(
  image: HTMLImageElement | ImageBitmap | HTMLCanvasElement | HTMLVideoElement,
): Promise<PetDetectionResult> {
  const model = await loadPetDetector();
  // mobilenet's TS types do not include ImageBitmap; paint it onto a canvas
  // when needed so the same code path works for HTMLImageElement and Bitmap.
  const renderable = toRenderable(image);
  const predictions = await model.classify(renderable, 50);

  let dogScore = 0;
  let catScore = 0;
  let topLabel = "";
  let topProb = 0;

  for (const p of predictions) {
    if (p.probability > topProb) {
      topProb = p.probability;
      topLabel = p.className;
    }
  }

  // The npm package only returns top-K class names + probabilities, not the
  // raw 1000-way distribution. To compute exact index-range sums we need the
  // raw tensor — use model.infer() and slice the relevant index ranges.
  const logits = model.infer(renderable, false) as import("@tensorflow/tfjs").Tensor;
  try {
    const squeezed = logits.squeeze();
    const size = squeezed.size;
    // MobileNet v2 from tfjs-models emits [1, 1001]: index 0 is "background",
    // indices 1..1000 map to ImageNet classes 0..999. Older v1 builds emit
    // [1, 1000]. Strip the background slot when present so our index ranges
    // line up with the canonical ImageNet ordering.
    const aligned =
      size === 1001 ? squeezed.slice([1], [1000]) : squeezed;

    const probs = (await aligned.data()) as Float32Array;

    // Some MobileNet variants emit logits, others emit post-softmax probs.
    // Detect by summing — if total ≈ 1 it is already probabilities.
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

  const petScore = dogScore + catScore;

  let kind: PetDetectionKind;
  if (petScore < PET_SCORE_THRESHOLD) {
    kind = "neither";
  } else {
    kind = dogScore >= catScore ? "dog" : "cat";
  }

  return { kind, dogScore, catScore, petScore, topLabel, topProb };
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
