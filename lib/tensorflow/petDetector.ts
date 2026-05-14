/**
 * OOD (out-of-distribution) pet detector built on top of MobileNet ImageNet.
 *
 * The binary cat-vs-dog head only knows cats and dogs, so we gate with
 * MobileNet (118 dog + 5 domestic cat ImageNet classes).
 *
 * Multi-region scanning: full frame, center, edges, corners, and a tight
 * center crop so small or off-center pets (selfies, held kittens) still
 * register somewhere. The orchestrator in `predict.ts` uses the strongest
 * region for MobileNet dog/cat *split* and can ensemble the binary head
 * across several crops.
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

export type RegionScore = {
  name: string;
  element: Renderable;
  dogScore: number;
  catScore: number;
  petScore: number;
  topLabel: string;
  topProb: number;
};

export type PetDetectionResult = {
  kind: PetDetectionKind;
  dogScore: number;
  catScore: number;
  petScore: number;
  topLabel: string;
  topProb: number;
  bestRegion: Renderable;
  bestRegionName: string;
  /** max(dog+cat) over all scanned regions — used for the OOD gate. */
  globalMaxPetScore: number;
  /** All regions with scores, sorted by petScore descending (for ensemble / diagnostics). */
  rankedRegions: RegionScore[];
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
 * Scans many crops, ranks them by pet mass, and returns the best crop plus
 * the full ranking for downstream ensemble binary runs.
 */
export async function detectPet(
  image: HTMLImageElement | ImageBitmap | HTMLCanvasElement | HTMLVideoElement,
): Promise<PetDetectionResult> {
  const model = await loadPetDetector();
  const baseRenderable = toRenderable(image);

  const regions = buildRegions(baseRenderable);
  const scored: RegionScore[] = [];

  for (const region of regions) {
    const s = await scoreImage(model, region.element);
    scored.push({
      name: region.name,
      element: region.element,
      dogScore: s.dogScore,
      catScore: s.catScore,
      petScore: s.dogScore + s.catScore,
      topLabel: s.topLabel,
      topProb: s.topProb,
    });
  }

  scored.sort((a, b) => b.petScore - a.petScore);
  const best = scored[0]!;
  const globalMaxPetScore = best.petScore;

  const kind: PetDetectionKind =
    globalMaxPetScore <= 0
      ? "neither"
      : best.dogScore >= best.catScore
        ? "dog"
        : "cat";

  return {
    kind,
    dogScore: best.dogScore,
    catScore: best.catScore,
    petScore: best.petScore,
    topLabel: best.topLabel,
    topProb: best.topProb,
    bestRegion: best.element,
    bestRegionName: best.name,
    globalMaxPetScore,
    rankedRegions: scored,
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
    { name: "center-tight", x: 0.25, y: 0.25, w: 0.5, h: 0.5 },
    { name: "top", x: 0, y: 0, w: 1, h: 0.7 },
    { name: "bottom", x: 0, y: 0.3, w: 1, h: 0.7 },
    { name: "left", x: 0, y: 0, w: 0.7, h: 1 },
    { name: "right", x: 0.3, y: 0, w: 0.7, h: 1 },
    { name: "tl", x: 0, y: 0, w: 0.55, h: 0.55 },
    { name: "tr", x: 0.45, y: 0, w: 0.55, h: 0.55 },
    { name: "bl", x: 0, y: 0.45, w: 0.55, h: 0.55 },
    { name: "br", x: 0.45, y: 0.45, w: 0.55, h: 0.55 },
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
