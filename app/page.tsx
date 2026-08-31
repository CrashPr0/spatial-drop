import type { Metadata } from "next";
import { AppleScanPreview } from "./components/AppleScanPreview";

export const metadata: Metadata = {
  title: "Spatial Drop — SJSU XR Prototyping Lab",
  description:
    "Preview an Apple Object Capture USDZ in the browser and place it with 8th Wall WebAR.",
};

export default function Home() {
  return <AppleScanPreview />;
}
