type Props = {
  label?: string;
  /** Tone variant — defaults to neutral; "brand" uses the navy accent. */
  variant?: "neutral" | "brand";
};

/**
 * Paw-print loader: four little paw silhouettes bounce in sequence.
 * Used both during model warm-up and during a classification.
 */
export function LoadingSpinner({ label = "Analyzing…", variant = "neutral" }: Props) {
  const dotColor =
    variant === "brand"
      ? "fill-brand"
      : "fill-neutral-800";

  return (
    <div
      className="flex flex-col items-center justify-center gap-3 px-2 py-6 text-neutral-600"
      role="status"
      aria-live="polite"
    >
      <div className="flex items-end gap-2">
        {[0, 1, 2, 3].map((i) => (
          <Paw
            key={i}
            className={`h-5 w-5 ${dotColor} animate-paw-bounce`}
            style={{ animationDelay: `${i * 0.15}s` }}
          />
        ))}
      </div>
      <span className="max-w-[min(100%,20rem)] text-center text-sm font-medium tracking-tight sm:max-w-none">
        {label}
      </span>
    </div>
  );
}

function Paw({
  className,
  style,
}: {
  className?: string;
  style?: React.CSSProperties;
}) {
  return (
    <svg
      viewBox="0 0 24 24"
      className={className}
      style={style}
      aria-hidden
    >
      {/* central pad */}
      <ellipse cx="12" cy="16" rx="4.5" ry="3.6" />
      {/* toes */}
      <ellipse cx="6.5" cy="11" rx="1.8" ry="2.4" />
      <ellipse cx="10" cy="7.5" rx="1.8" ry="2.4" />
      <ellipse cx="14" cy="7.5" rx="1.8" ry="2.4" />
      <ellipse cx="17.5" cy="11" rx="1.8" ry="2.4" />
    </svg>
  );
}
