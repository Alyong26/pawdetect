/**
 * Pet detector built on COCO-SSD object detection.
 *
 * Why COCO-SSD and not MobileNet ImageNet:
 *   - COCO-SSD is purpose-built object detection. "cat" and "dog" are two of
 *     the 80 primary COCO classes — the model learned to detect them as
 *     objects with bounding boxes, not as 2-of-1000 ImageNet labels.
 *   - One forward pass returns all detections, so we don't need an 11-region
 *     scan to find off-center pets. The detector finds them natively.
 *   - It's strict by design: if no "cat" or "dog" object is detected with
 *     reasonable confidence, the photo is rejected as "Not a Dog or Cat".
 *     The previous MobileNet pipeline summed dog-class probability mass,
 *     which could drift above the gate on non-pet photos.
 *
 * Model choice: `lite_mobilenet_v2` base — ~6 MB (smaller than the previous
 * ~14 MB MobileNet alpha 1.0), single-pass inference, suitable for browsers
 * and PWAs.
 */
import "@tensorflow/tfjs";
import type {
  ObjectDetection,
  DetectedObject,
} from "@tensorflow-models/coco-ssd";

export type PetDetectionKind = "dog" | "cat" | "neither";

export type PetDetectionResult = {
  kind: PetDetectionKind;
  /** Highest COCO-SSD detection score for "dog" in this image (0 if none). */
  dogScore: number;
  /** Highest COCO-SSD detection score for "cat" in this image (0 if none). */
  catScore: number;
  /** max(dogScore, catScore) — used for the OOD gate in `predict.ts`. */
  petScore: number;
  /**
   * The bounding box of the winning detection, cropped to a canvas with a
   * little padding around it. Fed to the user's trained binary head so it
   * sees a pet-centric input (instead of a full frame with the pet in a
   * corner). Falls back to the full image when no detection passes the gate.
   */
  bestRegion: HTMLCanvasElement;
  /** "dog" | "cat" | "full" — for debug / future UI. */
  bestRegionName: string;
};

let cocoSsdPromise: Promise<ObjectDetection> | null = null;

export async function loadPetDetector(): Promise<ObjectDetection> {
  if (!cocoSsdPromise) {
    cocoSsdPromise = (async () => {
      const cocoSsd = await import("@tensorflow-models/coco-ssd");
      return cocoSsd.load({ base: "lite_mobilenet_v2" });
    })().catch((error) => {
      cocoSsdPromise = null;
      throw error;
    });
  }
  return cocoSsdPromise;
}

/**
 * Run COCO-SSD over the image and pick the strongest cat / dog detection.
 *
 * `maxNumBoxes` 20 is far more than needed (even crowded photos rarely have
 * more than a few pets); the second arg is a minimum-score threshold that
 * COCO-SSD applies internally before non-max suppression. 0.2 lets us see
 * dog/cat detections that we'll filter ourselves with the stricter PET_GATE
 * in `predict.ts`.
 */
export async function detectPet(
  image: HTMLImageElement | ImageBitmap | HTMLCanvasElement | HTMLVideoElement,
): Promise<PetDetectionResult> {
  const model = await loadPetDetector();
  const renderable = toRenderable(image);

  const detections = await model.detect(renderable, 20, 0.2);

  let bestDog: DetectedObject | null = null;
  let bestCat: DetectedObject | null = null;
  for (const d of detections) {
    if (d.class === "dog") {
      if (!bestDog || d.score > bestDog.score) bestDog = d;
    } else if (d.class === "cat") {
      if (!bestCat || d.score > bestCat.score) bestCat = d;
    }
  }

  const dogScore = bestDog?.score ?? 0;
  const catScore = bestCat?.score ?? 0;
  const petScore = Math.max(dogScore, catScore);

  const kind: PetDetectionKind =
    petScore <= 0 ? "neither" : dogScore >= catScore ? "dog" : "cat";

  const winnerBbox =
    kind === "dog" ? bestDog?.bbox : kind === "cat" ? bestCat?.bbox : null;

  const bestRegion = winnerBbox
    ? cropBboxToCanvas(renderable, winnerBbox, 0.15)
    : toCanvas(renderable);
  const bestRegionName = kind === "neither" ? "full" : kind;

  return { kind, dogScore, catScore, petScore, bestRegion, bestRegionName };
}

/**
 * Crop a COCO-SSD bbox `[x, y, w, h]` from the image with `padRatio` padding
 * on every side, then square it so the binary head's 224×224 resize doesn't
 * have to crop or letterbox.
 */
function cropBboxToCanvas(
  image: HTMLImageElement | HTMLCanvasElement | HTMLVideoElement,
  bbox: number[],
  padRatio: number,
): HTMLCanvasElement {
  const { width: imgW, height: imgH } = naturalSize(image);
  const [bx, by, bw, bh] = bbox;

  const padX = bw * padRatio;
  const padY = bh * padRatio;
  let x = Math.max(0, bx - padX);
  let y = Math.max(0, by - padY);
  let w = Math.min(imgW - x, bw + 2 * padX);
  let h = Math.min(imgH - y, bh + 2 * padY);

  const side = Math.max(w, h);
  x = Math.max(0, x - (side - w) / 2);
  y = Math.max(0, y - (side - h) / 2);
  w = Math.min(imgW - x, side);
  h = Math.min(imgH - y, side);

  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.floor(w));
  canvas.height = Math.max(1, Math.floor(h));
  const ctx = canvas.getContext("2d");
  if (ctx) ctx.drawImage(image, x, y, w, h, 0, 0, canvas.width, canvas.height);
  return canvas;
}

function toCanvas(
  image: HTMLImageElement | HTMLCanvasElement | HTMLVideoElement,
): HTMLCanvasElement {
  const { width, height } = naturalSize(image);
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, width);
  canvas.height = Math.max(1, height);
  const ctx = canvas.getContext("2d");
  if (ctx) ctx.drawImage(image, 0, 0);
  return canvas;
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
