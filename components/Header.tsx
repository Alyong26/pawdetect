import Image from "next/image";
import Link from "next/link";

export function Header() {
  return (
    <header className="border-b border-neutral-200 bg-white/80 backdrop-blur supports-[backdrop-filter]:bg-white/70">
      <div className="mx-auto flex max-w-3xl items-center px-4 py-4 sm:px-6 sm:py-5">
        <Link
          href="/"
          aria-label="PawDetect home"
          className="inline-flex items-center transition-transform duration-300 hover:scale-[1.02]"
        >
          <Image
            src="/brand/pawdetect-logo.png"
            alt="PawDetect"
            width={260}
            height={78}
            priority
            className="h-14 w-auto sm:h-16"
          />
        </Link>
      </div>
    </header>
  );
}
