# PawDetect

Classify cats and dogs from any photo, in your browser.

PawDetect is a Progressive Web App that runs a MobileNetV2-based image classifier directly in your browser using [TensorFlow.js](https://www.tensorflow.org/js). It produces one of three results: **Dog**, **Cat**, or **Not a Dog or Cat** — no uploads, no accounts, no backend.

**Live:** https://pawdetect.vercel.app

## What it does

- Accepts a JPEG, PNG, WebP, or GIF image from your device.
- Resizes to 224×224, normalizes with MobileNet V2 preprocessing, and runs a sigmoid head with horizontal-flip test-time augmentation.
- Returns **Dog** or **Cat** when confidence is at least 70%; otherwise returns **Not a Dog or Cat**.
- Installable as a PWA on supported browsers.

## How it works

| Stage | Detail |
|-------|--------|
| Model | MobileNetV2 (ImageNet) + GAP + Dropout + sigmoid head, trained on a cats-vs-dogs dataset. |
| Runtime | TensorFlow.js with WebGL backend (falls back to CPU). |
| Preprocessing | 224×224 bilinear resize, MobileNet V2 normalization (`pixel/127.5 - 1`). |
| Inference | Two forward passes (original + horizontal flip), averaged. |
| Unknown class | If the higher of P(dog) or P(cat) is below 70%, the result is **Not a Dog or Cat**. |
| Privacy | Everything happens in the browser. Images never leave the device. |

## Run locally

```bash
npm install
npm run dev
```

Then open [http://localhost:3000](http://localhost:3000).

If the page or model assets behave oddly after pulling updates:

```bash
npm run dev:clean   # wipes .next and starts dev
```

## Production build

```bash
npm run build
npm start
```

## Project structure

```
app/                     Next.js App Router (layout, page, client shell loader)
components/              UI components (Header, Footer, ImageUploader, PredictionCard, ...)
lib/tensorflow/          loadModel, preprocess, predict, infer config
public/model/            model.json, group1-shard.bin, infer.json
colab/                   Colab training script for retraining MobileNetV2
scripts/                 Maintenance scripts (clean .next, generate icons, test loader)
```

## Replacing the model

The repository ships a trained MobileNetV2 export. To retrain:

1. Open `colab/pawdetect_colab_train.py` on Colab with a GPU runtime.
2. Upload your dataset (folders `cats/` and `dogs/` under one root).
3. Run with `PAWDETECT_DATA=/path/to/dataset python pawdetect_colab_train.py`.
4. Download `pawdetect_tfjs.zip`, unzip into `public/model/`, overwriting `model.json`, `group1-shard.bin`, and `infer.json`.
5. Hard-refresh once so the browser drops the cached model.

The loader (`lib/tensorflow/loadModel.ts`) applies Keras 3 → TensorFlow.js compatibility patches at runtime, including flattening any nested `Functional` blocks and renaming `DepthwiseConv2D` kernel weights.

## Deployment

This is a static Next.js 15 app. Deploy on Vercel:

```bash
npx vercel --prod
```

No environment variables are required.

## License

&copy; 2026 PawDetect. All rights reserved.
