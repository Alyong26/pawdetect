"use client";

import dynamic from "next/dynamic";

const PawDetectShell = dynamic(() => import("@/components/PawDetectShell"), {
  ssr: false,
  loading: () => (
    <div className="flex min-h-dvh items-center justify-center bg-slate-50 text-slate-600">
      <p className="text-sm">Loading PawDetect…</p>
    </div>
  ),
});

export default function HomePageClient() {
  return <PawDetectShell />;
}
