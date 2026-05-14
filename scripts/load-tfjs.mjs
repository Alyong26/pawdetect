/**
 * Prefer `@tensorflow/tfjs-node` (native TensorFlow); fall back to `@tensorflow/tfjs`
 * + CPU backend when the native addon is missing or cannot compile (common on Windows).
 */
export async function loadTf() {
  try {
    const tf = await import("@tensorflow/tfjs-node");
    await tf.ready();
    return tf;
  } catch {
    const tf = await import("@tensorflow/tfjs");
    await import("@tensorflow/tfjs-backend-cpu");
    await tf.setBackend("cpu");
    await tf.ready();
    return tf;
  }
}
