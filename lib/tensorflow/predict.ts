import * as tf from "@tensorflow/tfjs";
import { rawSigmoidToDogProbability } from "./inferConfig";
import { detectPet, type PetDetectionResult, type Renderable } from "./petDetector";
import { preprocessFromPixels, preprocessImageElement, type DecodedImageSource } from "./preprocess";

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
  /** Reason the photo was rejected. */
  unknownReason?: "ood" | "low-confidence";
  /** Raw OOD scores for diagnostics. */
  petDetector: PetDetectionResult;
};

/**
 * Combined dog + cat probability mass on ImageNet (across 6 regions) needed
 * for a photo to qualify as a pet candidate.
 *
 * The trained binary head is intentionally NOT consulted for OOD because it
 * has no notion of "neither" and is empirically biased toward Dog on non-pet
 * inputs (people, lions, food, objects, etc.).
 */
const PET_GATE = 0.12;

/**
 * After fusion, the winning class needs at least this much fused probability
 * for us to commit. Otherwise we report Unknown for safety.
 */
const FUSED_CONFIDENCE_FLOOR = 0.6;

/**
 * Weight of MobileNet (generalist) when fusing with the user-trained binary
 * head (specialist).
 *
 * 0.45 / 0.55 favours the binary head slightly because it is specifically
 * trained for cat-vs-dog discrimination — particularly useful now that we
 * feed it the pet-dominated crop (not the full frame). MobileNet still gets
 * close-to-equal weight so it can override the binary head on unusual breeds
 * where the binary head wavers (e.g., fluffy white Maltese → cat 59%).
 */
const MOBILENET_WEIGHT = 0.45;

/**
 * Pipeline:
 *   1. MobileNet OOD pass scans 6 regions (full, center, top, bottom, left,
 *      right) and returns the region with the strongest pet signal.
 *   2. petScore < PET_GATE → Unknown. Binary head not invoked.
 *   3. Otherwise run the user-trained binary head with horizontal-flip TTA
 *      on the BEST REGION — so it sees a pet-dominated patch instead of a
 *      person-dominated full frame.
 *   4. Fuse MobileNet's normalised dog/cat split (from the best region) with
 *      the binary head's probabilities.
 *   5. If the winning fused probability is below FUSED_CONFIDENCE_FLOOR,
 *      report Unknown; otherwise commit to Dog or Cat.
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

  // Run the trained binary head on the same region MobileNet identified as
  // most pet-dominated. This is the key fix for the dark-kitten-with-person
  // case: the binary head no longer has to fight with a dominant person in
  // the frame.
  const binary = await runBinaryHeadOnRegion(model, petDetector.bestRegion, options?.tta);

  // Normalise MobileNet's dog/cat scores within the pet budget so they live
  // on the same [0,1] scale as the binary head probabilities.
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

async function runBinaryHeadOnRegion(
  model: tf.LayersModel,
  region: Renderable,
  tta?: boolean,
): Promise<{ rawSigmoid: number; dogProbability: number; catProbability: number }> {
  // preprocessFromPixels handles HTMLCanvasElement / HTMLImageElement /
  // HTMLVideoElement via tf.browser.fromPixels.
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
