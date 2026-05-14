import Image from "next/image";

type Props = {
  label?: string;
};

/**
 * Full-bleed branded loader used while the app shell or model is warming up.
 * Logo breathes gently with a paw-trail ring underneath.
 */
export function BrandLoader({ label = "Loading PawDetect…" }: Props) {
  return (
    <div className="flex min-h-dvh flex-col items-center justify-center bg-neutral-50 px-6">
      <div className="flex flex-col items-center gap-6">
        <div className="relative">
          <div
            aria-hidden
            className="absolute inset-0 -m-4 rounded-full bg-brand/10 blur-xl animate-breath"
          />
          <Image
            src="/brand/pawdetect-logo.png"
            alt="PawDetect"
            width={220}
            height={66}
            priority
            className="relative h-16 w-auto animate-breath"
          />
        </div>
        <div className="flex items-end gap-2">
          {[0, 1, 2, 3].map((i) => (
            <svg
              key={i}
              viewBox="0 0 24 24"
              className="h-4 w-4 fill-brand animate-paw-bounce"
              style={{ animationDelay: `${i * 0.15}s` }}
              aria-hidden
            >
              <ellipse cx="12" cy="16" rx="4.5" ry="3.6" />
              <ellipse cx="6.5" cy="11" rx="1.8" ry="2.4" />
              <ellipse cx="10" cy="7.5" rx="1.8" ry="2.4" />
              <ellipse cx="14" cy="7.5" rx="1.8" ry="2.4" />
              <ellipse cx="17.5" cy="11" rx="1.8" ry="2.4" />
            </svg>
          ))}
        </div>
        <p className="text-sm font-medium text-neutral-500">{label}</p>
      </div>
    </div>
  );
}
