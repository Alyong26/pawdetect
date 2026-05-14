type Props = {
  label?: string;
};

export function LoadingSpinner({ label = "Analyzing…" }: Props) {
  return (
    <div className="flex items-center justify-center gap-3 py-6 text-neutral-600">
      <div
        className="h-5 w-5 rounded-full border-2 border-neutral-200 border-t-neutral-900 animate-spin"
        aria-hidden
      />
      <span className="text-sm">{label}</span>
    </div>
  );
}
