import * as tf from "@tensorflow/tfjs";
import { rawSigmoidToDogProbability } from "./inferConfig";
import { detectPet, type PetDetectionResult } from "./petDetector";
import { preprocessImageElement, type DecodedImageSource } from "./preprocess";

export type ClassLabel = "Dog" | "Cat" | "Not a Dog or Cat";

export type PredictionOutput = {
  /** Final classification including the Unknown class. */
  label: ClassLabel;
  /** Raw sigmoid from the binary head (≈ P(Keras class 1)). 0 when binary head was skipped. */
  rawSigmoid: number;
  /** Probability of Dog after folder-order mapping. */
  dogProbability: number;
  /** Probability of Cat (complement). */
  catProbability: number;
  /** Confidence for the displayed label, 0–100. */
  confidencePercent: number;
  /** True if the photo was classified as Not a Dog or Cat. */
  isUnknown: boolean;
  /** Reason the photo was rejected (only set when isUnknown is true). */
  unknownReason?: "ood" | "low-confidence";
  /** Raw OOD scores for diagnostics. */
  petDetector: PetDetectionResult;
};

/**
 * If MobileNet's combined dog + cat probability mass is ≥ this, we treat the
 * photo as definitely a pet and let the binary head pick the label.
 */
const PET_SCORE_STRICT = 0.18;

/**
 * Below this score the photo has essentially no ImageNet pet signal across
 * either the full-frame or the center-crop pass — reject without consulting
 * the binary head (it has no training-data support for this image).
 */
const PET_SCORE_FLOOR = 0.025;

/**
 * Between FLOOR and STRICT we let the binary head act as a tiebreaker. If it
 * commits with at least this much probability we accept the photo. This is
 * what rescues real pet photos where MobileNet is distracted by the framing
 * (selfies, occlusion, multi-subject, busy backgrounds).
 */
const BORDERLINE_BINARY_FLOOR = 0.7;

/**
 * Even on strong MobileNet signal, refuse to commit if the binary head is in
 * the 50–60% coin-flip zone.
 */
const BINARY_CONFIDENCE_FLOOR = 0.55;

/**
 * Top-level classifier.
 *
 * Flow:
 *   1. MobileNet OOD pass (`detectPet`) computes petScore from full + crop.
 *   2. petScore < FLOOR → Unknown (clearly not a pet).
 *   3. Otherwise run the binary cat-vs-dog head with horizontal-flip TTA.
 *   4. If petScore is strong (≥ STRICT) OR borderline but binary head agrees
 *      strongly (≥ BORDERLINE_BINARY_FLOOR), accept the binary head's pick.
 *   5. If the binary head is on the fence (< BINARY_CONFIDENCE_FLOOR), return
 *      Unknown for safety.
 */
export async function runPrediction(
  model: tf.LayersModel,
  image: DecodedImageSource,
  options?: { tta?: boolean },
): Promise<PredictionOutput> {
  const petDetector = await detectPet(image);

  if (petDetector.petScore < PET_SCORE_FLOOR) {
    return makeUnknown(petDetector, "ood");
  }

  const binary = await runBinaryHead(model, image, options?.tta);
  const topProb = Math.max(binary.dogProbability, binary.catProbability);
  const binaryLabel: "Dog" | "Cat" =
    binary.dogProbability >= 0.5 ? "Dog" : "Cat";

  const strongPetSignal = petDetector.petScore >= PET_SCORE_STRICT;
  const borderlineButBinaryConfident =
    petDetector.petScore >= PET_SCORE_FLOOR && topProb >= BORDERLINE_BINARY_FLOOR;

  if (!strongPetSignal && !borderlineButBinaryConfident) {
    return makeUnknown(petDetector, "ood", binary);
  }

  if (topProb < BINARY_CONFIDENCE_FLOOR) {
    return makeUnknown(petDetector, "low-confidence", binary);
  }

  return {
    label: binaryLabel,
    rawSigmoid: binary.rawSigmoid,
    dogProbability: binary.dogProbability,
    catProbability: binary.catProbability,
    confidencePercent: toPercent(topProb),
    isUnknown: false,
    petDetector,
  };
}

async function runBinaryHead(
  model: tf.LayersModel,
  image: DecodedImageSource,
  tta?: boolean,
): Promise<{ rawSigmoid: number; dogProbability: number; catProbability: number }> {
  const input = preprocessImageElement(image);
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
  // Confidence shown to the user expresses how confident we are this is NOT a
  // pet. Use the OOD signal as the primary source; fall back to binary head
  // margin when we have it.
  const notPetConfidence = clamp01(1 - petDetector.petScore);
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
