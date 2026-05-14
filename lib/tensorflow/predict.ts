import * as tf from "@tensorflow/tfjs";
import { rawSigmoidToDogProbability } from "./inferConfig";
import { detectPet, type PetDetectionResult } from "./petDetector";
import { preprocessImageElement, type DecodedImageSource } from "./preprocess";

export type ClassLabel = "Dog" | "Cat" | "Not a Dog or Cat";

export type PredictionOutput = {
  /** Final classification including the Unknown class. */
  label: ClassLabel;
  /** Raw sigmoid from the trained binary head (≈ P(Keras class 1)). 0 when skipped. */
  rawSigmoid: number;
  /** ENSEMBLE probability of Dog (trained head + MobileNet, confidence-weighted). */
  dogProbability: number;
  /** ENSEMBLE probability of Cat. */
  catProbability: number;
  /** Probability of Dog from ONLY the user's trained binary head. */
  binaryDogProbability: number;
  /** Probability of Cat from ONLY the user's trained binary head. */
  binaryCatProbability: number;
  /** Confidence for the displayed label, 0–100. */
  confidencePercent: number;
  /** True if the photo was classified as Not a Dog or Cat. */
  isUnknown: boolean;
  /** Reason the photo was rejected (only set when isUnknown is true). */
  unknownReason?: "ood" | "low-confidence";
  /** OOD scores (full + center crop) from MobileNet. */
  petDetector: PetDetectionResult;
};

/** MobileNet pet-score thresholds — see petDetector.ts for the rationale. */
const PET_SCORE_STRICT = 0.18;
const PET_SCORE_FLOOR = 0.025;
const BORDERLINE_BINARY_FLOOR = 0.7;
const ENSEMBLE_CONFIDENCE_FLOOR = 0.55;

/**
 * Top-level classifier.
 *
 * Order of operations:
 *   1. MobileNet OOD pass on full image + center crop (`detectPet`).
 *      Rejects photos with essentially no pet signal anywhere.
 *   2. The user's trained binary cat-vs-dog head (`runBinaryHead`) runs with
 *      horizontal-flip TTA. This is the project's primary classifier — it
 *      drives every accepted prediction.
 *   3. A confidence-aware ensemble (`ensembleDogProbability`) combines the
 *      trained head with MobileNet's dog/cat mass:
 *        - When the trained head is confident (≈ |p - 0.5| × 2 near 1.0),
 *          its weight is 0.8 and MobileNet is 0.2 — trained head essentially
 *          rules.
 *        - When the trained head is near the decision boundary (50–60%),
 *          its weight drops to 0.4 and MobileNet's 0.6 takes over as a
 *          tiebreaker.
 *      This preserves the project requirement that the trained model is
 *      always used, while fixing the failure mode where its uncertain
 *      verdicts produced visibly wrong labels.
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
  const binaryTop = Math.max(binary.dogProbability, binary.catProbability);

  const strongPetSignal = petDetector.petScore >= PET_SCORE_STRICT;
  const borderlineButBinaryConfident =
    petDetector.petScore >= PET_SCORE_FLOOR && binaryTop >= BORDERLINE_BINARY_FLOOR;

  if (!strongPetSignal && !borderlineButBinaryConfident) {
    return makeUnknown(petDetector, "ood", binary);
  }

  const ensembleDogP = ensembleDogProbability(
    binary.dogProbability,
    petDetector.dogScore,
    petDetector.catScore,
  );
  const ensembleCatP = 1 - ensembleDogP;
  const ensembleTop = Math.max(ensembleDogP, ensembleCatP);
  const finalLabel: "Dog" | "Cat" = ensembleDogP >= 0.5 ? "Dog" : "Cat";

  if (ensembleTop < ENSEMBLE_CONFIDENCE_FLOOR) {
    return makeUnknown(petDetector, "low-confidence", binary);
  }

  return {
    label: finalLabel,
    rawSigmoid: binary.rawSigmoid,
    dogProbability: ensembleDogP,
    catProbability: ensembleCatP,
    binaryDogProbability: binary.dogProbability,
    binaryCatProbability: binary.catProbability,
    confidencePercent: toPercent(ensembleTop),
    isUnknown: false,
    petDetector,
  };
}

/**
 * Confidence-weighted ensemble of the user's trained head and MobileNet.
 *
 * Returns P(Dog).
 *
 * - `binaryDogP`     : P(Dog) from the trained head
 * - `mobileDogScore` : ImageNet probability mass on dog classes
 * - `mobileCatScore` : ImageNet probability mass on cat classes
 *
 * Weights are adaptive:
 *   wBinary = 0.4 + 0.4 × binaryConfidence    (0.4 … 0.8)
 *   wMobile = 1 − wBinary                     (0.6 … 0.2)
 * where binaryConfidence = |binaryDogP − 0.5| × 2  ∈ [0, 1].
 */
function ensembleDogProbability(
  binaryDogP: number,
  mobileDogScore: number,
  mobileCatScore: number,
): number {
  const mobileMass = mobileDogScore + mobileCatScore;
  const mobileDogP = mobileMass > 0 ? mobileDogScore / mobileMass : 0.5;

  const binaryConfidence = Math.min(1, Math.abs(binaryDogP - 0.5) * 2);
  const wBinary = 0.4 + 0.4 * binaryConfidence;
  const wMobile = 1 - wBinary;

  return clamp01(wBinary * binaryDogP + wMobile * mobileDogP);
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
  const notPetConfidence = clamp01(1 - petDetector.petScore);
  return {
    label: "Not a Dog or Cat",
    rawSigmoid: binary?.rawSigmoid ?? 0,
    dogProbability: binary?.dogProbability ?? petDetector.dogScore,
    catProbability: binary?.catProbability ?? petDetector.catScore,
    binaryDogProbability: binary?.dogProbability ?? 0,
    binaryCatProbability: binary?.catProbability ?? 0,
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
