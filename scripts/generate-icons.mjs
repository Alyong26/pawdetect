/**
 * Generate the PWA + favicon assets from the canonical brand logo at
 * `public/brand/pawdetect-logo.png`.
 *
 * Outputs:
 *   public/icons/icon-192.png      (PWA, manifest)
 *   public/icons/icon-512.png      (PWA, manifest, maskable)
 *   app/icon.png                   (Next.js auto favicon, 64x64)
 *   app/apple-icon.png             (Next.js Apple touch icon, 180x180)
 *
 * Runs on postinstall, so contributors do not need to ship binaries around manually.
 */
import { access, mkdir } from "node:fs/promises";
import path from "node:path";
import sharp from "sharp";

const root = process.cwd();
const sourceLogo = path.join(root, "public", "brand", "pawdetect-logo.png");
const iconsDir = path.join(root, "public", "icons");
const appDir = path.join(root, "app");

const BACKGROUND = { r: 255, g: 255, b: 255, alpha: 1 };

async function exists(p) {
  try {
    await access(p);
    return true;
  } catch {
    return false;
  }
}

async function writeFromLogo(size, outPath, { padding = 0 } = {}) {
  const inner = Math.max(1, size - padding * 2);
  const resized = await sharp(sourceLogo)
    .resize(inner, inner, {
      fit: "contain",
      background: BACKGROUND,
    })
    .png()
    .toBuffer();
  await sharp({
    create: {
      width: size,
      height: size,
      channels: 4,
      background: BACKGROUND,
    },
  })
    .composite([{ input: resized, gravity: "center" }])
    .png()
    .toFile(outPath);
}

async function main() {
  if (!(await exists(sourceLogo))) {
    console.warn(`[icons] Brand logo missing at ${sourceLogo}; skipping icon generation.`);
    return;
  }

  await mkdir(iconsDir, { recursive: true });

  await writeFromLogo(192, path.join(iconsDir, "icon-192.png"), { padding: 8 });
  await writeFromLogo(512, path.join(iconsDir, "icon-512.png"), { padding: 16 });
  await writeFromLogo(64, path.join(appDir, "icon.png"));
  await writeFromLogo(180, path.join(appDir, "apple-icon.png"), { padding: 6 });

  console.log("[icons] Generated PawDetect icons from brand logo.");
}

await main();
