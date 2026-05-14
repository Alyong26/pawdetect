/**
 * One-off (re-runnable): take a logo whose source has a baked-in black
 * background, chroma-key the black to alpha, and write the canonical
 * `public/brand/pawdetect-logo.png` used everywhere in the app.
 *
 * The logo art itself uses dark navy (B channel ≈ 120) so anything with a
 * maximum channel < ~30 is genuinely background. We feather the boundary
 * to avoid jagged edges from JPEG compression artifacts.
 *
 * Usage: node scripts/strip-logo-bg.mjs <input> <output>
 */
import sharp from "sharp";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");

const input = process.argv[2] ?? path.join(root, "public", "brand", "pawdetect-logo-v2-source.png");
const output = process.argv[3] ?? path.join(root, "public", "brand", "pawdetect-logo.png");

const SOLID_BG_MAX = 14;
const FULL_FG_MIN = 32;

const { data, info } = await sharp(input)
  .ensureAlpha()
  .raw()
  .toBuffer({ resolveWithObject: true });

if (info.channels !== 4) {
  throw new Error(`Expected 4 channels (RGBA) after ensureAlpha, got ${info.channels}`);
}

for (let i = 0; i < data.length; i += 4) {
  const r = data[i];
  const g = data[i + 1];
  const b = data[i + 2];
  const m = Math.max(r, g, b);

  let alpha;
  if (m <= SOLID_BG_MAX) {
    alpha = 0;
  } else if (m >= FULL_FG_MIN) {
    alpha = 255;
  } else {
    alpha = Math.round(((m - SOLID_BG_MAX) * 255) / (FULL_FG_MIN - SOLID_BG_MAX));
  }
  data[i + 3] = alpha;
}

await sharp(data, {
  raw: { width: info.width, height: info.height, channels: 4 },
})
  .png({ compressionLevel: 9 })
  .toFile(output);

console.log(`[strip-logo-bg] wrote ${output} (${info.width}×${info.height})`);
