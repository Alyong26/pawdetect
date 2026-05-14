import Image from "next/image";

export function Header() {
  return (
    <header className="border-b border-neutral-200 bg-white">
      <div className="mx-auto flex max-w-3xl items-center gap-3 px-4 py-4 sm:px-6">
        <div className="relative h-8 w-8 overflow-hidden rounded-lg">
          <Image
            src="/icons/icon-192.png"
            alt="PawDetect"
            fill
            sizes="32px"
            className="object-cover"
            priority
          />
        </div>
        <span className="text-base font-semibold tracking-tight text-neutral-900">
          PawDetect
        </span>
      </div>
    </header>
  );
}
