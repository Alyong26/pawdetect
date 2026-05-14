import * as tf from "@tensorflow/tfjs";
import { rawSigmoidToDogProbability } from "./inferConfig";
import { detectPet, type PetDetectionResult } from "./petDetector";
import { preprocessImageElement, type DecodedImageSource } from "./preprocess";

export type ClassLabel = "Dog" | "Cat" | "Not a Dog or Cat";

export type PredictionOutput = {
  /** Final classification including the Unknown class. */
  label: ClassLabel;
  /** Raw sigmoid from the trained binary head (≈ P(class 1)). 0 if not invoked. */
  rawSigmoid: number;
  /** Final fused probability for Dog after combining MobileNet + binary head. */
  dogProbability: number;
  /** Final fused probability for Cat after combining MobileNet + binary head. */
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
 * Combined dog + cat probability mass on ImageNet (across full frame + center
 * crop) needed for a photo to even qualify as a pet candidate.
 *
 * IMPORTANT: the trained binary head is intentionally NOT consulted for OOD.
 * A binary cat-vs-dog classifier cannot say "neither" — when shown a person,
 * a lion, food, or a car it must commit to one side and is empirically biased
 * toward Dog with very high probability. Using its confidence as a tiebreaker
 * for borderline OOD photos is what caused the previous false-positive wave
 * (people, a lion, cannabis buds → "Dog"). So the OOD gate is decided by
 * MobileNet alone.
 */
const PET_GATE = 0.15;

/**
 * After fusing MobileNet's dog/cat distribution with the binary head, the
 * winning class needs at least this fused probability for us to commit. This
 * catches the rare case where MobileNet and the trained head genuinely
 * disagree.
 */
const FUSED_CONFIDENCE_FLOOR = 0.6;

/**
 * Weight given to MobileNet (generalist; 118 dog breeds + 5 cat classes) when
 * fusing with the user's trained binary head (specialist).
 *
 * Both are kept in the loop:
 *   - MobileNet is the source of truth for "is this a pet, and broadly which".
 *   - The binary head adjusts the verdict based on whatever dataset-specific
 *     features it learned.
 *
 * 0.55 gives MobileNet a slight edge, which empirically corrects miscalls on
 * unusual breeds (e.g., a fluffy white Maltese the binary head was tilting
 * toward Cat).
 */
const MOBILENET_WEIGHT = 0.55;

/**
 * Pipeline:
 *   1. MobileNet OOD pass (full frame + centered 70% crop, take max).
 *   2. petScore < PET_GATE → Unknown. Binary head not invoked.
 *   3. Otherwise run the user-trained binary head with horizontal-flip TTA.
 *   4. Fuse MobileNet's normalised dog/cat split with the binary head's
 *      probabilities (weighted average, weights sum to 1).
 *   5. If the winning fused probability is below FUSED_CONFIDENCE_FLOOR,
 *      report Unknown for safety; otherwise commit to Dog or Cat.
 */
export async function runPrediction(
  model: tf.LayersModel,
  image: DecodedImageSource,
  options?: { tta?: boolean },
): Promise<PredictionOutput> {
  const petDetector = await detectPet(image);

  if (petDetector.petScore < PET_GATE) {
    return makeUnknown(petDetector, "ood");
  }

  const binary = await runBinaryHead(model, image, options?.tta);

  // Normalise MobileNet's dog/cat scores within the pet budget so they live on
  // the same [0,1] scale as the binary head probabilities.
  const m = Math.max(petDetector.petScore, 1e-6);
  const dogM = petDetector.dogScore / m;
  const catM = petDetector.catScore / m;

  const dogFused =
    MOBILENET_WEIGHT * dogM + (1 - MOBILENET_WEIGHT) * binary.dogProbability;
  const catFused =
    MOBILENET_WEIGHT * catM + (1 - MOBILENET_WEIGHT) * binary.catProbability;

  const topProb = Math.max(dogFused, catFused);
  if (topProb < FUSED_CONFIDENCE_FLOOR) {
    return makeUnknown(petDetector, "low-confidence", binary);
  }

  const label: "Dog" | "Cat" = dogFused >= catFused ? "Dog" : "Cat";

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
