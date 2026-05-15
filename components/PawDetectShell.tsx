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
  loadPetDetector,
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
        // Kick off the ImageNet pet detector warmup in the background so the
        // user does not pay its download cost on the first classify click.
        void loadPetDetector().catch(() => {
          /* if this fails, we retry lazily during classify */
        });
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
        const result = await runPrediction(model, image);
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
    <div className="relative isolate flex min-h-dvh min-w-0 flex-col bg-neutral-50">
      {/* Ambient backdrop — subtle brand-tinted glow behind the hero. */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-x-0 top-0 -z-10 h-[min(480px,100svh)] bg-gradient-to-b from-brand/[0.045] via-white/40 to-transparent"
      />
      <div
        aria-hidden
        className="pointer-events-none absolute left-1/2 top-20 -z-10 h-[min(320px,55vw)] w-[min(100vw,820px)] max-w-full -translate-x-1/2 rounded-full bg-brand/5 blur-3xl sm:top-28"
      />

      <Header />
      <main className="flex min-h-0 min-w-0 flex-1">
        <section className="mx-auto w-full min-w-0 max-w-3xl px-safe pt-10 pb-12 sm:pt-16 sm:pb-16 md:max-w-4xl md:pt-20">
          <div className="animate-fade-in-up text-balance text-center">
            <h1 className="text-[1.65rem] font-semibold leading-tight tracking-tight text-neutral-900 sm:text-4xl md:text-5xl">
              Classify cats and dogs
              <br className="hidden sm:block" />{" "}
              <span className="text-brand">from any photo.</span>
            </h1>
            <p className="mx-auto mt-3 max-w-xl px-1 text-sm leading-relaxed text-neutral-600 sm:mt-4 sm:text-base md:max-w-2xl">
              Upload an image and PawDetect will tell you if it&apos;s a dog, a cat,
              or something else — running entirely in your browser.
            </p>
          </div>

          <div
            className="mt-8 space-y-5 animate-fade-in-up sm:mt-12 sm:space-y-6"
            style={{ animationDelay: "80ms" }}
          >
            <ImageUploader disabled={disabled} onFileSelected={handleFile} />

            {previewUrl ? (
              <div className="animate-scale-in overflow-hidden rounded-2xl border border-neutral-200/60 bg-white shadow-elev">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={previewUrl}
                  alt="Selected preview"
                  className="mx-auto max-h-[min(70svh,22rem)] w-full object-contain sm:max-h-80 md:max-h-[28rem]"
                />
              </div>
            ) : null}

            <button
              type="button"
              onClick={handlePredict}
              disabled={!previewUrl || disabled}
              className="group inline-flex min-h-[48px] w-full touch-manipulation items-center justify-center gap-2 rounded-xl bg-neutral-900 px-4 py-3.5 text-sm font-semibold text-white shadow-elev transition-colors duration-200 hover:bg-brand hover:shadow-lg active:bg-brand/90 disabled:cursor-not-allowed disabled:bg-neutral-300 disabled:shadow-none md:min-h-[52px] md:text-base lg:hover:-translate-y-0.5 lg:disabled:translate-y-0 motion-reduce:lg:hover:translate-y-0"
            >
              {predicting
                ? "Analyzing…"
                : modelStatus === "loading"
                  ? "Loading model…"
                  : "Classify image"}
            </button>

            {userError ? (
              <p className="animate-fade-in rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm leading-snug text-red-700 break-words">
                {userError}
              </p>
            ) : null}
            {modelError ? (
              <p className="animate-fade-in rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm leading-snug text-amber-800 break-words">
                {modelError}
              </p>
            ) : null}

            {predicting || modelStatus === "loading" ? (
              <LoadingSpinner
                variant="brand"
                label={modelStatus === "loading" ? "Loading model…" : "Analyzing image…"}
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
