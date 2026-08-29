"use client";

import { createElement, useEffect } from "react";

type ModelViewerProps = {
  src: string;
  alt: string;
  placement?: "floor" | "wall";
  resizable?: boolean;
  className?: string;
  reveal?: "auto" | "interaction" | "manual";
};

export function ModelViewer({
  src,
  alt,
  placement = "floor",
  resizable = true,
  className,
  reveal = "auto",
}: ModelViewerProps) {
  useEffect(() => {
    void import("@google/model-viewer");
  }, []);

  return createElement(
    "model-viewer",
    {
      src,
      alt,
      class: className,
      ar: true,
      "ar-modes": "webxr scene-viewer quick-look",
      "ar-placement": placement,
      "ar-scale": resizable ? "auto" : "fixed",
      "camera-controls": true,
      "auto-rotate": true,
      autoplay: true,
      reveal,
      "shadow-intensity": "0.8",
      "environment-image": "neutral",
      "touch-action": "pan-y",
      loading: "eager",
    },
    createElement(
      "button",
      { slot: "ar-button", className: "ar-launch-button", type: "button" },
      createElement("span", { className: "button-mark", "aria-hidden": true }, "✦"),
      "View in your space",
    ),
  );
}
