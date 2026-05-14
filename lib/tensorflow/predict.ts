import * as tf from "@tensorflow/tfjs";
import { rawSigmoidToDogProbability } from "./inferConfig";
import {
  detectPet,
  type PetDetectionResult,
  type Renderable,
  type RegionScore,
} from "./petDetector";
import {
  preprocessFromPixels,
  preprocessImageElement,
  type DecodedImageSource,
} from "./preprocess";

export type ClassLabel = "Dog" | "Cat" | "Not a Dog or Cat";

export type PredictionOutput = {
  /** Final classification including the Unknown class. */
  label: ClassLabel;
  /** Raw sigmoid from the trained binary head on the primary crop (≈ P(class 1)). */
  rawSigmoid: number;
  /** Final probability for Dog after ensemble + fusion / overrides. */
  dogProbability: number;
  /** Final probability for Cat. */
  catProbability: number;
  /** Confidence for the displayed label, 0–100. */
  confidencePercent: number;
  isUnknown: boolean;
  unknownReason?: "ood" | "low-confidence";
  petDetector: PetDetectionResult;
};

/**
 * OOD gate on the strongest crop anywhere in the frame. Slightly below 0.12
 * because we now add corner + tight-center crops that lift true pets without
 * lifting random non-pet photos much (they stay near 0 everywhere).
 */
const PET_GATE = 0.09;

const FUSED_CONFIDENCE_FLOOR = 0.58;

const MOBILENET_WEIGHT = 0.42;

/**
 * When the specialist binary head is confident "Dog" but MobileNet's
 * domestic dog vs domestic cat mass on the *best* crop still leans cat,
 * trust MobileNet — typical failure mode: black kittens, funny poses, or
 * human arms + fur confusing the binary head. Symmetric rule for "Cat"
 * vs dog-leaning MobileNet (e.g. some terrier faces).
 */
const DISAGREE_BINARY_MIN = 0.54;
const DISAGREE_MOB_MARGIN = 0.045;

export async function runPrediction(
  model: tf.LayersModel,
  image: DecodedImageSource,
  options?: { tta?: boolean },
): Promise<PredictionOutput> {
  const petDetector = await detectPet(image);

  if (petDetector.globalMaxPetScore < PET_GATE) {
    return makeUnknown(petDetector, "ood");
  }

  const binary = await ensembleBinaryHead(model, petDetector.rankedRegions, options?.tta);

  const best = petDetector.rankedRegions[0]!;
  const m = Math.max(best.petScore, 1e-6);
  const dogM = best.dogScore / m;
  const catM = best.catScore / m;

  let dogFused =
    MOBILENET_WEIGHT * dogM + (1 - MOBILENET_WEIGHT) * binary.dogProbability;
  let catFused =
    MOBILENET_WEIGHT * catM + (1 - MOBILENET_WEIGHT) * binary.catProbability;

  let label: "Dog" | "Cat" =
    dogFused >= catFused ? "Dog" : "Cat";

  // Disagreement resolver (see module comment).
  if (
    binary.dogProbability >= DISAGREE_BINARY_MIN &&
    catM > dogM + DISAGREE_MOB_MARGIN
  ) {
    label = "Cat";
    catFused = Math.max(catFused, binary.dogProbability);
    dogFused = 1 - catFused;
  } else if (
    binary.catProbability >= DISAGREE_BINARY_MIN &&
    dogM > catM + DISAGREE_MOB_MARGIN
  ) {
    label = "Dog";
    dogFused = Math.max(dogFused, binary.catProbability);
    catFused = 1 - dogFused;
  }

  const topProb = Math.max(dogFused, catFused);
  if (topProb < FUSED_CONFIDENCE_FLOOR) {
    return makeUnknown(petDetector, "low-confidence", binary);
  }

  return {
    label,
    rawSigmoid: binary.rawSigmoid,
    dogProbability: dogFused,
    catProbability: catFused,
    confidencePercent: toPercent(topProb),
    isUnknown: false,
    petDetector,
  };
}

/**
 * Average the binary head over the full frame plus up to two other strongest
 * crops so one bad crop (mostly skin / clothing) cannot dominate.
 */
async function ensembleBinaryHead(
  model: tf.LayersModel,
  ranked: RegionScore[],
  tta?: boolean,
): Promise<{
  rawSigmoid: number;
  dogProbability: number;
  catProbability: number;
}> {
  const picks: RegionScore[] = [];
  const full = ranked.find((r) => r.name === "full");
  if (full) picks.push(full);
  for (const r of ranked) {
    if (picks.some((p) => p.name === r.name)) continue;
    if (r.petScore < 0.035) continue;
    picks.push(r);
    if (picks.length >= 3) break;
  }
  if (picks.length === 0 && ranked[0]) picks.push(ranked[0]);

  let sumDog = 0;
  let sumRaw = 0;
  for (const r of picks) {
    const b = await runBinaryHeadOnRegion(model, r.element, tta);
    sumDog += b.dogProbability;
    sumRaw += b.rawSigmoid;
  }
  const n = picks.length;
  const dogProbability = sumDog / n;
  const catProbability = 1 - dogProbability;
  const rawSigmoid = sumRaw / n;

  return { rawSigmoid, dogProbability, catProbability };
}

async function runBinaryHeadOnRegion(
  model: tf.LayersModel,
  region: Renderable,
  tta?: boolean,
): Promise<{ rawSigmoid: number; dogProbability: number; catProbability: number }> {
  const input =
    region instanceof HTMLImageElement
      ? preprocessImageElement(region)
      : preprocessFromPixels(region as HTMLCanvasElement | HTMLVideoElement);
  const useTta = tta !== false;

  const sigmoid = useTta
    ? tf.tidy(() => {
        const p1 = model.predict(input) as tf.Tensor;
        const flipped = tf.image.flipLeftRight(input) as tf.Tensor4D;
        const p2 = model.predict(flipped) as tf.Tensor;
        return tf.add(p1, p2).div(2);
      })
    : (model.predict(input) as tf.Tensor);

  const data = await sigmoid.data();
  const rawSigmoid = Number(data[0]);
  const dogProbability = rawSigmoidToDogProbability(rawSigmoid);
  const catProbability = 1 - dogProbability;

  sigmoid.dispose();
  input.dispose();

  return { rawSigmoid, dogProbability, catProbability };
}

function makeUnknown(
  petDetector: PetDetectionResult,
  reason: "ood" | "low-confidence",
  binary?: { rawSigmoid: number; dogProbability: number; catProbability: number },
): PredictionOutput {
  const notPetConfidence = clamp01(1 - petDetector.globalMaxPetScore);
  return {
    label: "Not a Dog or Cat",
    rawSigmoid: binary?.rawSigmoid ?? 0,
    dogProbability: binary?.dogProbability ?? petDetector.dogScore,
    catProbability: binary?.catProbability ?? petDetector.catScore,
    confidencePercent: toPercent(notPetConfidence),
    isUnknown: true,
    unknownReason: reason,
    petDetector,
  };
}

function toPercent(p: number): number {
  return Math.round(clamp01(p) * 1000) / 10;
}

function clamp01(p: number): number {
  return Math.min(1, Math.max(0, p));
}
