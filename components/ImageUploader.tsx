"use client";

import { useCallback, useId, useRef, useState } from "react";

type Props = {
  disabled?: boolean;
  onFileSelected: (file: File) => void;
};

export function ImageUploader({ disabled, onFileSelected }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const inputId = useId();
  const [isDragging, setIsDragging] = useState(false);

  const handleDrop = useCallback(
    (event: React.DragEvent<HTMLLabelElement>) => {
      event.preventDefault();
      setIsDragging(false);
      if (disabled) return;
      const file = event.dataTransfer.files?.[0];
      if (file) onFileSelected(file);
    },
    [disabled, onFileSelected],
  );

  return (
    <label
      htmlFor={inputId}
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      className={[
        "group flex cursor-pointer flex-col items-center justify-center rounded-2xl border border-dashed px-6 py-14 text-center transition-all duration-300",
        isDragging
          ? "scale-[1.01] border-brand bg-brand/[0.04] shadow-elev"
          : "border-neutral-300 bg-white/80 backdrop-blur",
        "hover:-translate-y-0.5 hover:border-brand hover:bg-white hover:shadow-elev",
        disabled ? "pointer-events-none opacity-50" : "",
      ].join(" ")}
    >
      <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-full bg-brand/[0.06] text-brand transition-colors duration-300 group-hover:bg-brand/[0.1]">
        <svg
          className="h-6 w-6"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.75"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden
        >
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="17 8 12 3 7 8" />
          <line x1="12" y1="3" x2="12" y2="15" />
        </svg>
      </div>
      <p className="text-sm font-semibold text-neutral-900">Upload an image</p>
      <p className="mt-1 text-xs text-neutral-500">
        Drag and drop or click to choose · JPEG, PNG, WebP, GIF
      </p>
      <input
        id={inputId}
        ref={inputRef}
        type="file"
        accept="image/*"
        className="sr-only"
        disabled={disabled}
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) onFileSelected(file);
          event.target.value = "";
        }}
      />
    </label>
  );
}
