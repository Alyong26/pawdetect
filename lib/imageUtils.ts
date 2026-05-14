const IMAGE_MIME_TYPES = new Set(["image/jpeg", "image/png", "image/webp", "image/gif"]);

export function isSupportedImageFile(file: File): boolean {
  if (IMAGE_MIME_TYPES.has(file.type)) return true;
  const lower = file.name.toLowerCase();
  return [".jpg", ".jpeg", ".png", ".webp", ".gif"].some((ext) => lower.endsWith(ext));
}

export function readFileAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("Could not read the selected file."));
    reader.onload = () => {
      if (typeof reader.result === "string") resolve(reader.result);
      else reject(new Error("Unexpected file reader result."));
    };
    reader.readAsDataURL(file);
  });
}

export function loadImageElement(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.crossOrigin = "anonymous";
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("The image could not be decoded. Try a different file."));
    image.src = src;
  });
}

/**
 * Decode for model input: applies JPEG/WebP EXIF orientation when supported so pixels match
 * what the user sees and training-time decoding (browser-oriented photos).
 */
export async function decodeImageForModel(dataUrl: string): Promise<HTMLImageElement | ImageBitmap> {
  if (typeof createImageBitmap === "undefined") {
    return loadImageElement(dataUrl);
  }
  try {
    const response = await fetch(dataUrl);
    const blob = await response.blob();
    return await createImageBitmap(blob, { imageOrientation: "from-image" });
  } catch {
    return loadImageElement(dataUrl);
  }
}
