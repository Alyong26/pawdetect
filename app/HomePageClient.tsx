"use client";

import dynamic from "next/dynamic";
import { BrandLoader } from "@/components/BrandLoader";

const PawDetectShell = dynamic(() => import("@/components/PawDetectShell"), {
  ssr: false,
  loading: () => <BrandLoader />,
});

export default function HomePageClient() {
  return <PawDetectShell />;
}
