import Image from "next/image";
import Link from "next/link";

export function Header() {
  return (
    <header className="border-b border-neutral-200 bg-white">
      <div className="mx-auto flex max-w-3xl items-center px-4 py-4 sm:px-6">
        <Link
          href="/"
          aria-label="PawDetect home"
          className="inline-flex items-center"
        >
          <Image
            src="/brand/pawdetect-logo.png"
            alt="PawDetect"
            width={160}
            height={48}
            priority
            className="h-10 w-auto"
          />
        </Link>
      </div>
    </header>
  );
}
