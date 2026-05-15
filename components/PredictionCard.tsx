import type { PredictionOutput } from "@/lib/tensorflow";

type Props = {
  result: PredictionOutput | null;
};

export function PredictionCard({ result }: Props) {
  if (!result) return null;

  const isPet = result.label === "Dog" || result.label === "Cat";

  return (
    <div className="animate-fade-in-up rounded-2xl border border-neutral-200/60 bg-white p-5 text-center shadow-elev sm:p-7 md:p-8">
      <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-neutral-400 sm:tracking-[0.2em]">
        Result
      </p>
      <p
        className={`mt-2 break-words text-balance text-3xl font-semibold leading-tight tracking-tight sm:mt-3 sm:text-4xl md:text-5xl ${
          isPet ? "text-brand" : "text-neutral-900"
        }`}
      >
        {result.label}
      </p>

      {isPet ? (
        <>
          <p className="mt-3 text-sm text-neutral-500 sm:mt-4">
            Confidence{" "}
            <span className="font-semibold text-neutral-900">
              {result.confidencePercent.toFixed(1)}%
            </span>
          </p>
          <div className="mx-auto mt-3 h-1.5 w-full max-w-xs overflow-hidden rounded-full bg-neutral-100 sm:mt-4 sm:w-56 md:max-w-sm">
            <div
              className="h-full rounded-full bg-gradient-to-r from-brand to-brand-accent transition-[width] duration-700 ease-out"
              style={{ width: `${Math.min(100, Math.max(4, result.confidencePercent))}%` }}
              aria-hidden
            />
          </div>
        </>
      ) : (
        <p className="mx-auto mt-3 max-w-sm text-pretty px-1 text-sm leading-relaxed text-neutral-500 sm:mt-4 md:max-w-md">
          PawDetect couldn&apos;t find a cat or dog in this photo. Try a clearer
          picture of a real cat or dog.
        </p>
      )}
    </div>
  );
}
