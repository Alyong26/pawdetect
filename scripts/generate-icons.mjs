import { mkdir } from "node:fs/promises";
import path from "node:path";
import sharp from "sharp";

const root = process.cwd();
const iconsDir = path.join(root, "public", "icons");

const sky = { r: 14, g: 165, b: 233 };

async function writeIcon(size, filename) {
  const filePath = path.join(iconsDir, filename);
  await sharp({
    create: {
      width: size,
      height: size,
      channels: 3,
      background: sky,
    },
  })
    .png()
    .toFile(filePath);
}

await mkdir(iconsDir, { recursive: true });
await writeIcon(192, "icon-192.png");
await writeIcon(512, "icon-512.png");

console.log("Generated PWA icons in public/icons");
