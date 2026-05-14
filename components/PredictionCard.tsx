import type { PredictionOutput } from "@/lib/tensorflow";

type Props = {
  result: PredictionOutput | null;
};

export function PredictionCard({ result }: Props) {
  if (!result) return null;

  const isPet = result.label === "Dog" || result.label === "Cat";
  const tone = isPet ? "text-neutral-900" : "text-neutral-500";

  return (
    <div className="rounded-2xl border border-neutral-200 bg-white p-6 text-center">
      <p className="text-xs font-medium uppercase tracking-wider text-neutral-500">
        Result
      </p>
      <p className={`mt-2 text-3xl font-semibold tracking-tight ${tone}`}>
        {result.label}
      </p>
      <p className="mt-3 text-sm text-neutral-500">
        Confidence{" "}
        <span className="font-medium text-neutral-900">
          {result.confidencePercent.toFixed(1)}%
        </span>
      </p>
    </div>
  );
}
