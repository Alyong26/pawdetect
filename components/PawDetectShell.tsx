"use client";

import * as tf from "@tensorflow/tfjs";
import { useCallback, useEffect, useState } from "react";
import { Footer } from "@/components/Footer";
import { Header } from "@/components/Header";
import { ImageUploader } from "@/components/ImageUploader";
import { LoadingSpinner } from "@/components/LoadingSpinner";
import { PredictionCard } from "@/components/PredictionCard";
import { decodeImageForModel, isSupportedImageFile, readFileAsDataUrl } from "@/lib/imageUtils";
import {
  loadModel,
  preprocessImageElement,
  runPrediction,
  type PredictionOutput,
} from "@/lib/tensorflow";

type ModelStatus = "idle" | "loading" | "ready" | "error";

export default function PawDetectShell() {
  const [modelStatus, setModelStatus] = useState<ModelStatus>("loading");
  const [modelError, setModelError] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictionOutput | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [userError, setUserError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function warmup() {
      setModelStatus("loading");
      setModelError(null);
      try {
        const webglReady = await tf.setBackend("webgl");
        await tf.ready();
        if (!webglReady) {
          await tf.setBackend("cpu");
          await tf.ready();
        }
        await loadModel();
        if (!cancelled) setModelStatus("ready");
      } catch (error) {
        console.error(error);
        if (!cancelled) {
          setModelStatus("error");
          setModelError(
            error instanceof Error
              ? error.message
              : "Unable to load the model.",
          );
        }
      }
    }

    void warmup();
    return () => {
      cancelled = true;
    };
  }, []);

  const disabled = modelStatus !== "ready" || predicting;

  const handleFile = useCallback(async (file: File) => {
    setUserError(null);
    setPrediction(null);

    if (!isSupportedImageFile(file)) {
      setUserError("Please choose a JPEG, PNG, WebP, or GIF image.");
      return;
    }

    try {
      const dataUrl = await readFileAsDataUrl(file);
      setPreviewUrl((prev) => {
        if (prev?.startsWith("blob:")) URL.revokeObjectURL(prev);
        return dataUrl;
      });
    } catch {
      setUserError("Could not read that file. Try another image.");
    }
  }, []);

  const handlePredict = useCallback(async () => {
    if (!previewUrl || modelStatus !== "ready") return;
    setPredicting(true);
    setUserError(null);
    setPrediction(null);

    try {
      const model = await loadModel();
      const image = await decodeImageForModel(previewUrl);
      try {
        const tensor = preprocessImageElement(image);
        const result = await runPrediction(model, tensor);
        setPrediction(result);
      } finally {
        if ("close" in image && typeof (image as ImageBitmap).close === "function") {
          (image as ImageBitmap).close();
        }
      }
    } catch (error) {
      console.error(error);
      setUserError(
        error instanceof Error
          ? error.message
          : "Prediction failed. Try a different image.",
      );
    } finally {
      setPredicting(false);
    }
  }, [modelStatus, previewUrl]);

  return (
    <div className="flex min-h-dvh flex-col bg-neutral-50">
      <Header />
      <main className="flex-1">
        <section className="mx-auto w-full max-w-3xl px-4 pt-12 pb-10 sm:px-6 sm:pt-16">
          <div className="text-center">
            <h1 className="text-3xl font-semibold tracking-tight text-neutral-900 sm:text-4xl">
              Classify cats and dogs from any photo.
            </h1>
            <p className="mx-auto mt-3 max-w-xl text-sm text-neutral-600 sm:text-base">
              Upload an image and PawDetect will tell you if it&apos;s a dog, a cat, or
              something else — running entirely in your browser.
            </p>
          </div>

          <div className="mt-10 space-y-6">
            <ImageUploader disabled={disabled} onFileSelected={handleFile} />

            {previewUrl ? (
              <div className="overflow-hidden rounded-2xl border border-neutral-200 bg-white">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={previewUrl}
                  alt="Selected"
                  className="mx-auto max-h-80 w-full object-contain"
                />
              </div>
            ) : null}

            <button
              type="button"
              onClick={handlePredict}
              disabled={!previewUrl || disabled}
              className="inline-flex w-full items-center justify-center rounded-xl bg-neutral-900 px-4 py-3 text-sm font-medium text-white transition hover:bg-neutral-800 disabled:cursor-not-allowed disabled:bg-neutral-300"
            >
              {predicting
                ? "Analyzing…"
                : modelStatus === "loading"
                  ? "Loading model…"
                  : "Classify image"}
            </button>

            {userError ? (
              <p className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                {userError}
              </p>
            ) : null}
            {modelError ? (
              <p className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
                {modelError}
              </p>
            ) : null}

            {predicting || modelStatus === "loading" ? (
              <LoadingSpinner
                label={modelStatus === "loading" ? "Loading model…" : "Analyzing…"}
              />
            ) : (
              <PredictionCard result={prediction} />
            )}
          </div>
        </section>
      </main>
      <Footer />
    </div>
  );
}
