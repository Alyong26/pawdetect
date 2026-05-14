import * as tf from "@tensorflow/tfjs";
import { rawSigmoidToDogProbability } from "./inferConfig";
import { detectPet, type PetDetectionResult } from "./petDetector";
import {
  preprocessFromPixels,
  type DecodedImageSource,
} from "./preprocess";

export type ClassLabel = "Dog" | "Cat" | "Not a Dog or Cat";

export type PredictionOutput = {
  /** Final classification including the Unknown class. */
  label: ClassLabel;
  /** Raw sigmoid from the trained binary head, ≈ P(class 1) = P(Dog). 0 when not invoked. */
  rawSigmoid: number;
  /** Final probability for Dog after COCO-SSD + binary-head fusion. */
  dogProbability: number;
  /** Final probability for Cat after COCO-SSD + binary-head fusion. */
  catProbability: number;
  /** Confidence for the displayed label, 0–100. */
  confidencePercent: number;
  isUnknown: boolean;
  unknownReason?: "ood";
  petDetector: PetDetectionResult;
};

/**
 * Strict OOD gate. A photo must have at least one COCO-SSD "cat" or "dog"
 * detection with score >= this to be classified as a pet at all.
 *
 * 0.5 is intentionally strict: lions / wolves / vehicles / food / people /
 * plants reliably score well below 0.5 because COCO-SSD didn't learn to
 * detect them as cat-or-dog. Domestic cats and dogs (the classes the model
 * was trained to detect) regularly score >= 0.7.
 */
const PET_GATE = 0.5;

export async function runPrediction(
  model: tf.LayersModel,
  image: DecodedImageSource,
  options?: { tta?: boolean },
): Promise<PredictionOutput> {
  const petDetector = await detectPet(image);

  if (petDetector.petScore < PET_GATE) {
    return makeUnknown(petDetector);
  }

  // COCO-SSD has located the pet's bounding box. Run the user's trained binary
  // head on that crop so it sees pet-centric pixels instead of "person + tiny
  // pet in the corner". This is what makes the binary head's specialist
  // training pay off in selfies and held-pet photos.
  const binary = await runBinaryHead(model, petDetector.bestRegion, options?.tta);

  // COCO-SSD is the primary verdict. It's specifically trained to distinguish
  // cats vs dogs vs everything else and won't get fooled by the colour-bias
  // patterns the user's binary head sometimes learns on a small dataset.
  // The binary head only adjusts the displayed confidence — it cannot flip
  // the dog/cat verdict.
  const cocoVerdict: "Dog" | "Cat" = petDetector.kind === "dog" ? "Dog" : "Cat";
  const binaryVerdict: "Dog" | "Cat" =
    binary.dogProbability >= 0.5 ? "Dog" : "Cat";
  const agreement = cocoVerdict === binaryVerdict;

  const cocoConfidence = petDetector.petScore;
  const binaryConfidence =
    cocoVerdict === "Dog" ? binary.dogProbability : binary.catProbability;

  /**
   * Weighted confidence:
   *   - When COCO-SSD and the binary head agree, blend (65/35) so a
   *     confident binary head can push a 70% detection up toward 85%.
   *   - When they disagree, lean heavily on COCO-SSD (it caught the actual
   *     object) and just dampen confidence a bit so the result honestly
   *     reflects that the two signals disagree.
   */
  const finalConfidence = agreement
    ? clamp01(0.65 * cocoConfidence + 0.35 * binaryConfidence)
    : clamp01(Math.max(cocoConfidence * 0.9, 0.5));

  return {
    label: cocoVerdict,
    rawSigmoid: binary.rawSigmoid,
    dogProbability:
      cocoVerdict === "Dog" ? finalConfidence : 1 - finalConfidence,
    catProbability:
      cocoVerdict === "Cat" ? finalConfidence : 1 - finalConfidence,
    confidencePercent: toPercent(finalConfidence),
    isUnknown: false,
    petDetector,
  };
}

/**
 * Run the user's trained binary head on a single canvas (the pet crop).
 * TTA defaults to OFF — COCO-SSD already gives a strong primary signal, and
 * one extra forward pass per classify just slows things down.
 */
async function runBinaryHead(
  model: tf.LayersModel,
  region: HTMLCanvasElement,
  tta?: boolean,
): Promise<{
  rawSigmoid: number;
  dogProbability: number;
  catProbability: number;
}> {
  const input = preprocessFromPixels(region);
  const useTta = tta === true;

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

function makeUnknown(petDetector: PetDetectionResult): PredictionOutput {
  const notPetConfidence = clamp01(1 - petDetector.petScore);
  return {
    label: "Not a Dog or Cat",
    rawSigmoid: 0,
    dogProbability: petDetector.dogScore,
    catProbability: petDetector.catScore,
    confidencePercent: toPercent(notPetConfidence),
    isUnknown: true,
    unknownReason: "ood",
    petDetector,
  };
}

function toPercent(p: number): number {
  return Math.round(clamp01(p) * 1000) / 10;
}

function clamp01(p: number): number {
  return Math.min(1, Math.max(0, p));
}
