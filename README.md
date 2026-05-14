# PawDetect

Classify cats and dogs from any photo, in your browser.

PawDetect is a Progressive Web App that runs a MobileNetV2-based image classifier directly in your browser using [TensorFlow.js](https://www.tensorflow.org/js). It produces one of three results: **Dog**, **Cat**, or **Not a Dog or Cat** — no uploads, no accounts, no backend.

**Live:** https://pawdetect.vercel.app

## What it does

- Accepts a JPEG, PNG, WebP, or GIF image from your device.
- Runs a two-stage classifier in the browser and returns **Dog**, **Cat**, or **Not a Dog or Cat**.
- Installable as a PWA on supported browsers.

## How it works

The pipeline combines the project's **trained binary cat-vs-dog head** with a generic **MobileNet ImageNet** model for out-of-distribution rejection and as a confidence-aware tiebreaker.

| Stage | Detail |
|-------|--------|
| Trained binary head | MobileNetV2 (ImageNet) + GAP + Dropout + sigmoid, fine-tuned on a cats-vs-dogs dataset. Lives in `public/model/`. |
| Runtime | TensorFlow.js with WebGL backend (falls back to CPU). |
| Preprocessing | 224×224 bilinear resize, MobileNet V2 normalization (`pixel/127.5 − 1`). |
| OOD gate | MobileNet v2 alpha 0.5 (ImageNet, 1k classes) runs on the full image **and** a centered 70% crop. We sum probability mass on the 118 ImageNet dog classes (indices 151–268) and 5 cat classes (281–285). Below a small floor the photo is rejected as **Not a Dog or Cat**. |
| Binary inference | Two forward passes through the trained head (original + horizontal flip), averaged. |
| Ensemble | The trained head's `P(Dog)` is combined with MobileNet's dog-vs-cat ratio using confidence-adaptive weights: when the trained head is confident, its weight is up to 0.8; when it is near the decision boundary, MobileNet rises to 0.6 as a tiebreaker. This preserves the trained model as the primary classifier while preventing visibly wrong labels on borderline cases. |
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
app/                     Next.js App Router (layout, page, client shell loader, icon assets)
components/              UI components (Header, Footer, ImageUploader, BrandLoader, LoadingSpinner, PredictionCard, PawDetectShell)
lib/tensorflow/          loadModel, preprocess, predict, petDetector (OOD gate), inferConfig
public/model/            model.json, group1-shard.bin, infer.json (the trained binary head)
public/brand/            Canonical brand logo (source of truth for every icon size)
public/icons/            PWA icons generated from the brand logo
colab/                   Colab training script for retraining MobileNetV2
scripts/                 Maintenance scripts (clean .next, generate icons, train/export)
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
