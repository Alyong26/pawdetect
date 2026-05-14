import type { PredictionOutput } from "@/lib/tensorflow";

type Props = {
  result: PredictionOutput | null;
};

export function PredictionCard({ result }: Props) {
  if (!result) return null;

  const isPet = result.label === "Dog" || result.label === "Cat";

  return (
    <div className="animate-fade-in-up rounded-2xl border border-neutral-200 bg-white p-6 text-center shadow-sm">
      <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-neutral-500">
        Result
      </p>
      <p
        className={`mt-2 text-3xl font-semibold tracking-tight sm:text-4xl ${
          isPet ? "text-brand" : "text-neutral-900"
        }`}
      >
        {result.label}
      </p>

      {isPet ? (
        <>
          <p className="mt-3 text-sm text-neutral-500">
            Confidence{" "}
            <span className="font-semibold text-neutral-900">
              {result.confidencePercent.toFixed(1)}%
            </span>
          </p>
          <div className="mx-auto mt-3 h-1.5 w-48 overflow-hidden rounded-full bg-neutral-100">
            <div
              className="h-full rounded-full bg-brand transition-[width] duration-700"
              style={{ width: `${Math.min(100, Math.max(4, result.confidencePercent))}%` }}
              aria-hidden
            />
          </div>
        </>
      ) : (
        <p className="mx-auto mt-3 max-w-sm text-sm leading-relaxed text-neutral-500">
          PawDetect couldn&apos;t find a cat or dog in this photo. Try a clearer
          picture of a real cat or dog.
        </p>
      )}
    </div>
  );
}
