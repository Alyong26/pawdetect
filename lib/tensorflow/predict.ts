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
 * Binary-head margin floor. Even when MobileNet says "this is a pet",
 * we still require the cat-vs-dog head to commit with at least this much
 * probability. Below the floor we fall back to "Not a Dog or Cat" to avoid
 * coin-flip predictions like "Dog, 52%".
 */
const BINARY_CONFIDENCE_FLOOR = 0.6;

/**
 * Top-level classifier. Order of operations:
 *   1. Run MobileNet ImageNet on the decoded image to compute dog / cat
 *      probability mass (the OOD gate). This is what rejects cars, food,
 *      landscapes, etc.
 *   2. If the gate says "pet", run the binary cat-vs-dog head with TTA and
 *      use its (calibrated) sigmoid for the final label and confidence.
 *   3. If the binary head is on the fence (max prob < BINARY_CONFIDENCE_FLOOR),
 *      we still report Unknown for safety.
 */
export async function runPrediction(
  model: tf.LayersModel,
  image: DecodedImageSource,
  options?: { tta?: boolean },
): Promise<PredictionOutput> {
  const petDetector = await detectPet(image);

  const noBinary = {
    rawSigmoid: 0,
    dogProbability: petDetector.dogScore,
    catProbability: petDetector.catScore,
  };

  if (petDetector.kind === "neither") {
    // Show how confident we are that this is NOT a pet.
    const notPetConfidence = clamp01(1 - petDetector.petScore);
    return {
      ...noBinary,
      label: "Not a Dog or Cat",
      confidencePercent: toPercent(notPetConfidence),
      isUnknown: true,
      unknownReason: "ood",
      petDetector,
    };
  }

  // OOD gate accepted — run the binary head for the final cat-vs-dog call.
  const input = preprocessImageElement(image);
  const useTta = options?.tta !== false;

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

  const topProb = Math.max(dogProbability, catProbability);
  const petLabel: "Dog" | "Cat" = dogProbability >= 0.5 ? "Dog" : "Cat";

  if (topProb < BINARY_CONFIDENCE_FLOOR) {
    return {
      rawSigmoid,
      dogProbability,
      catProbability,
      label: "Not a Dog or Cat",
      confidencePercent: toPercent(clamp01(1 - petDetector.petScore)),
      isUnknown: true,
      unknownReason: "low-confidence",
      petDetector,
    };
  }

  return {
    rawSigmoid,
    dogProbability,
    catProbability,
    label: petLabel,
    confidencePercent: toPercent(topProb),
    isUnknown: false,
    petDetector,
  };
}

function toPercent(p: number): number {
  return Math.round(clamp01(p) * 1000) / 10;
}

function clamp01(p: number): number {
  return Math.min(1, Math.max(0, p));
}
