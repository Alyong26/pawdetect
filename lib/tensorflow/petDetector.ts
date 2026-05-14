/**
 * OOD (out-of-distribution) pet detector built on top of MobileNet ImageNet.
 *
 * The binary cat-vs-dog head in `public/model/` only knows cats and dogs.
 * Threshold its sigmoid all you want — it can still confidently misclassify a
 * person, a lion, food, etc. So we gate with MobileNet (1k ImageNet classes)
 * which DOES have a "neither" answer (anything outside the 118 dog + 5 cat
 * indices).
 *
 * Multi-region scanning. A single center-crop pass misses pets that are:
 *   - Off-center (cat to the side of a selfie)
 *   - Held in the lower half of the frame
 *   - Cropped at the top by a person above them
 * So we scan 6 regions and report the strongest one. The orchestrator in
 * `predict.ts` then feeds that exact region to the binary head, so the
 * trained model sees a pet-dominated patch instead of a person-dominated
 * full frame.
 */
import "@tensorflow/tfjs";
import type { MobileNet } from "@tensorflow-models/mobilenet";

/** ImageNet class index range for dog breeds (Chihuahua … Mexican hairless). */
const DOG_CLASS_MIN = 151;
const DOG_CLASS_MAX = 268;
/** ImageNet class indices for cat classes (tabby … Egyptian cat). */
const CAT_CLASS_INDICES = [281, 282, 283, 284, 285] as const;

export type PetDetectionKind = "dog" | "cat" | "neither";

/** A renderable surface that tfjs / MobileNet / canvas can all consume. */
export type Renderable =
  | HTMLImageElement
  | HTMLCanvasElement
  | HTMLVideoElement;

export type PetDetectionResult = {
  /** Coarse verdict — "neither" iff every region agreed there was no pet. */
  kind: PetDetectionKind;
  /** Probability mass on ImageNet dog classes in the strongest region. */
  dogScore: number;
  /** Probability mass on ImageNet cat classes in the strongest region. */
  catScore: number;
  /** dogScore + catScore — primary "is this a pet" signal. */
  petScore: number;
  /** Best top-1 ImageNet label in the strongest region. */
  topLabel: string;
  /** Probability of the top-1 label. */
  topProb: number;
  /**
   * The region with the highest pet signal. Use this as the binary head's
   * input so it sees a pet-dominated patch instead of the full frame.
   */
  bestRegion: Renderable;
  /** Human-readable region tag for diagnostics. */
  bestRegionName: string;
};

let mobilenetPromise: Promise<MobileNet> | null = null;

/** Loads MobileNet v2 (alpha 0.5) once and caches the promise. */
export async function loadPetDetector(): Promise<MobileNet> {
  if (!mobilenetPromise) {
    mobilenetPromise = (async () => {
      const mobilenet = await import("@tensorflow-models/mobilenet");
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

type Region = {
  name: string;
  element: Renderable;
};

/**
 * Scans the image at 6 regions (full frame, center, top, bottom, left, right)
 * and returns the strongest pet signal along with the canvas of that region.
 */
export async function detectPet(
  image: HTMLImageElement | ImageBitmap | HTMLCanvasElement | HTMLVideoElement,
): Promise<PetDetectionResult> {
  const model = await loadPetDetector();
  const baseRenderable = toRenderable(image);

  const regions = buildRegions(baseRenderable);

  let best: { region: Region; scores: PassScores } | null = null;

  for (const region of regions) {
    const scores = await scoreImage(model, region.element);
    if (!best || scores.dogScore + scores.catScore > best.scores.dogScore + best.scores.catScore) {
      best = { region, scores };
    }
  }

  // regions[0] is always the full frame, so `best` is guaranteed non-null.
  // Narrow for TS.
  const { region: bestRegion, scores: bestScores } = best!;

  const dogScore = bestScores.dogScore;
  const catScore = bestScores.catScore;
  const petScore = dogScore + catScore;

  const kind: PetDetectionKind =
    petScore <= 0 ? "neither" : dogScore >= catScore ? "dog" : "cat";

  return {
    kind,
    dogScore,
    catScore,
    petScore,
    topLabel: bestScores.topLabel,
    topProb: bestScores.topProb,
    bestRegion: bestRegion.element,
    bestRegionName: bestRegion.name,
  };
}

function buildRegions(image: Renderable): Region[] {
  const regions: Region[] = [{ name: "full", element: image }];
  const tries: Array<{
    name: string;
    x: number;
    y: number;
    w: number;
    h: number;
  }> = [
    { name: "center", x: 0.15, y: 0.15, w: 0.7, h: 0.7 },
    { name: "top", x: 0, y: 0, w: 1, h: 0.7 },
    { name: "bottom", x: 0, y: 0.3, w: 1, h: 0.7 },
    { name: "left", x: 0, y: 0, w: 0.7, h: 1 },
    { name: "right", x: 0.3, y: 0, w: 0.7, h: 1 },
  ];
  for (const t of tries) {
    const canvas = regionCrop(image, t.x, t.y, t.w, t.h);
    if (canvas) regions.push({ name: t.name, element: canvas });
  }
  return regions;
}

async function scoreImage(
  model: MobileNet,
  renderable: Renderable,
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
    // MobileNet v2 emits [1, 1001] (background + 1k); v1 emits [1, 1000].
    // Strip the background slot when present so our index ranges line up
    // with the canonical ImageNet ordering.
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
): Renderable {
  const isBitmap =
    typeof ImageBitmap !== "undefined" && image instanceof ImageBitmap;
  if (!isBitmap) {
    return image as Renderable;
  }
  const bitmap = image as ImageBitmap;
  const canvas = document.createElement("canvas");
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const ctx = canvas.getContext("2d");
  if (ctx) ctx.drawImage(bitmap, 0, 0);
  return canvas;
}

function regionCrop(
  image: Renderable,
  xRatio: number,
  yRatio: number,
  wRatio: number,
  hRatio: number,
): HTMLCanvasElement | null {
  const { width, height } = naturalSize(image);
  if (width < 32 || height < 32) return null;
  const w = Math.max(16, Math.floor(width * wRatio));
  const h = Math.max(16, Math.floor(height * hRatio));
  const x = Math.floor(width * xRatio);
  const y = Math.floor(height * yRatio);
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.drawImage(image, x, y, w, h, 0, 0, w, h);
  return canvas;
}

function naturalSize(image: Renderable): { width: number; height: number } {
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
