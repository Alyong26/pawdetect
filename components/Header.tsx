import Image from "next/image";
import Link from "next/link";

export function Header() {
  return (
    <header className="border-b border-neutral-200 bg-white/80 backdrop-blur supports-[backdrop-filter]:bg-white/70 pt-safe">
      <div className="mx-auto flex max-w-3xl items-center px-safe py-3 sm:py-4 md:max-w-4xl md:py-5">
        <Link
          href="/"
          aria-label="PawDetect home"
          className="inline-flex min-h-[44px] min-w-[44px] items-center transition-transform duration-300 hover:scale-[1.02] active:scale-[0.98] motion-reduce:transition-none motion-reduce:hover:scale-100"
        >
          <Image
            src="/brand/pawdetect-logo.png"
            alt="PawDetect"
            width={260}
            height={78}
            priority
            sizes="(max-width: 480px) 200px, (max-width: 768px) 240px, 260px"
            className="h-11 w-auto sm:h-14 md:h-16"
          />
        </Link>
      </div>
    </header>
  );
}
