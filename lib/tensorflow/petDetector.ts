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
 *
 * We use v2 alpha 1.0 (~14 MB, cached by the PWA service worker) instead of
 * the smaller alpha 0.5. The smaller model was confusing black cats with
 * dark-furred dog breeds (schipperke, black Labrador, Scottish terrier),
 * which left the disagreement resolver in `predict.ts` unable to fire.
 *
 * Scoring matches the canonical ImageNet label strings that `classify()`
 * returns, instead of summing tensor indices directly — this is robust to
 * the "background class" offset that the v2 model checkpoint includes.
 */
import "@tensorflow/tfjs";
import type { MobileNet } from "@tensorflow-models/mobilenet";

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

/** Loads MobileNet v2 (alpha 1.0) once and caches the promise. */
export async function loadPetDetector(): Promise<MobileNet> {
  if (!mobilenetPromise) {
    mobilenetPromise = (async () => {
      const mobilenet = await import("@tensorflow-models/mobilenet");
      // v2 alpha 1.0 224 — 3.4M params, ~14 MB. The PWA SW caches the weights
      // after first load, so it's a one-time download per user.
      return mobilenet.load({ version: 2, alpha: 1.0 });
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

/**
 * Score a single region by asking MobileNet for the full sorted distribution,
 * then summing the probability mass for the 5 canonical ImageNet cat labels
 * and the 118 canonical ImageNet dog breed labels — exact-string match against
 * `IMAGENET_CAT_LABELS` / `IMAGENET_DOG_LABELS`. Earlier we used `/\bcat\b/`
 * but that false-matched class 383 (`Madagascar cat` — a lemur) and class 387
 * (`lesser panda, ..., bear cat, cat bear, ...` — a panda), inflating
 * catScore on non-pet photos.
 */
async function scoreImage(
  model: MobileNet,
  renderable: Renderable,
): Promise<PassScores> {
  const predictions = await model.classify(renderable, 1000);

  let dogScore = 0;
  let catScore = 0;
  let topLabel = "";
  let topProb = 0;

  for (const p of predictions) {
    if (p.probability > topProb) {
      topProb = p.probability;
      topLabel = p.className;
    }
    const lower = p.className.toLowerCase();
    if (IMAGENET_CAT_LABELS.has(lower)) {
      catScore += p.probability;
    } else if (IMAGENET_DOG_LABELS.has(lower)) {
      dogScore += p.probability;
    }
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

/**
 * Canonical ImageNet labels for the 5 domestic-cat classes (indices 281–285),
 * exactly as `MobileNet.classify()` returns them after lowercasing.
 *
 * Intentionally excludes class 383 (`Madagascar cat`, a lemur) and 387
 * (`bear cat`, a panda) — they contain the word "cat" but are not cats.
 * Also excludes 286–293 (cougar / lynx / leopard / lion / tiger / cheetah)
 * which are wild cats, not the domestic pets this app classifies.
 */
const IMAGENET_CAT_LABELS = new Set<string>([
  "tabby, tabby cat",
  "tiger cat",
  "persian cat",
  "siamese cat, siamese",
  "egyptian cat",
]);

/**
 * Canonical ImageNet labels for the 118 dog classes (indices 151–268),
 * lowercased to match `MobileNet.classify()` output after we lowercase it.
 *
 * Intentionally excludes wolves / foxes / hyenas (269–280) and big cats
 * (286–293 — cougar / lynx / leopard / lion / tiger / cheetah etc.), so a
 * lion or wolf photo reports "Not a Dog or Cat", which is what users want
 * from a domestic-pet classifier.
 */
const IMAGENET_DOG_LABELS = new Set<string>([
  "chihuahua",
  "japanese spaniel",
  "maltese dog, maltese terrier, maltese",
  "pekinese, pekingese, peke",
  "shih-tzu",
  "blenheim spaniel",
  "papillon",
  "toy terrier",
  "rhodesian ridgeback",
  "afghan hound, afghan",
  "basset, basset hound",
  "beagle",
  "bloodhound, sleuthhound",
  "bluetick",
  "black-and-tan coonhound",
  "walker hound, walker foxhound",
  "english foxhound",
  "redbone",
  "borzoi, russian wolfhound",
  "irish wolfhound",
  "italian greyhound",
  "whippet",
  "ibizan hound, ibizan podenco",
  "norwegian elkhound, elkhound",
  "otterhound, otter hound",
  "saluki, gazelle hound",
  "scottish deerhound, deerhound",
  "weimaraner",
  "staffordshire bullterrier, staffordshire bull terrier",
  "american staffordshire terrier, staffordshire terrier, american pit bull terrier, pit bull terrier",
  "bedlington terrier",
  "border terrier",
  "kerry blue terrier",
  "irish terrier",
  "norfolk terrier",
  "norwich terrier",
  "yorkshire terrier",
  "wire-haired fox terrier",
  "lakeland terrier",
  "sealyham terrier, sealyham",
  "airedale, airedale terrier",
  "cairn, cairn terrier",
  "australian terrier",
  "dandie dinmont, dandie dinmont terrier",
  "boston bull, boston terrier",
  "miniature schnauzer",
  "giant schnauzer",
  "standard schnauzer",
  "scotch terrier, scottish terrier, scottie",
  "tibetan terrier, chrysanthemum dog",
  "silky terrier, sydney silky",
  "soft-coated wheaten terrier",
  "west highland white terrier",
  "lhasa, lhasa apso",
  "flat-coated retriever",
  "curly-coated retriever",
  "golden retriever",
  "labrador retriever",
  "chesapeake bay retriever",
  "german short-haired pointer",
  "vizsla, hungarian pointer",
  "english setter",
  "irish setter, red setter",
  "gordon setter",
  "brittany spaniel",
  "clumber, clumber spaniel",
  "english springer, english springer spaniel",
  "welsh springer spaniel",
  "cocker spaniel, english cocker spaniel, cocker",
  "sussex spaniel",
  "irish water spaniel",
  "kuvasz",
  "schipperke",
  "groenendael",
  "malinois",
  "briard",
  "kelpie",
  "komondor",
  "old english sheepdog, bobtail",
  "shetland sheepdog, shetland sheep dog, shetland",
  "collie",
  "border collie",
  "bouvier des flandres, bouviers des flandres",
  "rottweiler",
  "german shepherd, german shepherd dog, german police dog, alsatian",
  "doberman, doberman pinscher",
  "miniature pinscher",
  "greater swiss mountain dog",
  "bernese mountain dog",
  "appenzeller",
  "entlebucher",
  "boxer",
  "bull mastiff",
  "tibetan mastiff",
  "french bulldog",
  "great dane",
  "saint bernard, st bernard",
  "eskimo dog, husky",
  "malamute, malemute, alaskan malamute",
  "siberian husky",
  "dalmatian, coach dog, carriage dog",
  "affenpinscher, monkey pinscher, monkey dog",
  "basenji",
  "pug, pug-dog",
  "leonberg",
  "newfoundland, newfoundland dog",
  "great pyrenees",
  "samoyed, samoyede",
  "pomeranian",
  "chow, chow chow",
  "keeshond",
  "brabancon griffon",
  "pembroke, pembroke welsh corgi",
  "cardigan, cardigan welsh corgi",
  "toy poodle",
  "miniature poodle",
  "standard poodle",
  "mexican hairless",
]);
