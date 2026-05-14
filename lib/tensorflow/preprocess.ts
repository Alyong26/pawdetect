import * as tf from "@tensorflow/tfjs";
import { getActivePixelNorm } from "./inferConfig";

export type DecodedImageSource = HTMLImageElement | ImageBitmap;

/**
 * Converts a decoded image into a batched float32 tensor shaped [1, 224, 224, 3].
 * Normalization follows `/model/infer.json` `pixelNorm` (see Colab `write_infer_json`).
 */
export function preprocessImageElement(image: DecodedImageSource): tf.Tensor4D {
  return tf.tidy(() => {
    const pixels = tf.browser.fromPixels(image).toFloat();
    const resized = tf.image.resizeBilinear(pixels, [224, 224]);
    const clipped = resized.clipByValue(0, 255);
    const mode = getActivePixelNorm();
    const normalized =
      mode === "mobilenet_v2" ? clipped.div(127.5).sub(1.0) : clipped.div(255.0);
    return normalized.expandDims(0) as tf.Tensor4D;
  });
}

/**
 * Convenience helper when you already have a decoded bitmap (e.g. from canvas).
 */
export function preprocessFromPixels(
  pixelData: ImageData | HTMLCanvasElement | HTMLVideoElement,
): tf.Tensor4D {
  return tf.tidy(() => {
    const pixels = tf.browser.fromPixels(pixelData).toFloat();
    const resized = tf.image.resizeBilinear(pixels, [224, 224]);
    const clipped = resized.clipByValue(0, 255);
    const mode = getActivePixelNorm();
    const normalized =
      mode === "mobilenet_v2" ? clipped.div(127.5).sub(1.0) : clipped.div(255.0);
    return normalized.expandDims(0) as tf.Tensor4D;
  });
}
