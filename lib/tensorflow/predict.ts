import * as tf from "@tensorflow/tfjs";
import { rawSigmoidToDogProbability } from "./inferConfig";

export type ClassLabel = "Dog" | "Cat" | "Not a Dog or Cat";

export type PredictionOutput = {
  /** Final classification including the Unknown class. */
  label: ClassLabel;
  /** Raw sigmoid from the model (≈ P(Keras class 1)). */
  rawSigmoid: number;
  /** Probability of Dog after folder-order mapping. */
  dogProbability: number;
  /** Probability of Cat (complement). */
  catProbability: number;
  /** Confidence for the chosen pet label, 0–100 (always max(dog, cat)). */
  confidencePercent: number;
  /** True if confidence falls below the Unknown threshold. */
  isUnknown: boolean;
};

/**
 * Confidence required (per-class probability) to commit to Dog or Cat.
 * Below this we report "Not a Dog or Cat" — useful when the input is neither
 * (the binary head can't say "unknown" by itself, so we use margin-from-0.5).
 */
const PET_CONFIDENCE_THRESHOLD = 0.7;

/**
 * Runs a forward pass on the preprocessed batch tensor, applies horizontal-flip
 * TTA by default, and returns Dog / Cat / "Not a Dog or Cat" based on confidence.
 */
export async function runPrediction(
  model: tf.LayersModel,
  input: tf.Tensor4D,
  options?: { tta?: boolean },
): Promise<PredictionOutput> {
  const useTta = options?.tta !== false;

  const logits = useTta
    ? tf.tidy(() => {
        const p1 = model.predict(input) as tf.Tensor;
        const flipped = tf.image.flipLeftRight(input) as tf.Tensor4D;
        const p2 = model.predict(flipped) as tf.Tensor;
        return tf.add(p1, p2).div(2);
      })
    : (model.predict(input) as tf.Tensor);

  const data = await logits.data();
  const rawSigmoid = Number(data[0]);
  const dogProbability = rawSigmoidToDogProbability(rawSigmoid);
  const catProbability = 1 - dogProbability;

  logits.dispose();
  input.dispose();

  const topProb = Math.max(dogProbability, catProbability);
  const isUnknown = topProb < PET_CONFIDENCE_THRESHOLD;
  const petLabel: "Dog" | "Cat" = dogProbability >= 0.5 ? "Dog" : "Cat";
  const label: ClassLabel = isUnknown ? "Not a Dog or Cat" : petLabel;

  return {
    label,
    rawSigmoid,
    dogProbability,
    catProbability,
    confidencePercent: Math.round(Math.min(1, Math.max(0, topProb)) * 1000) / 10,
    isUnknown,
  };
}
