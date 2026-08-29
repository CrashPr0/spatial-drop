import type { Metadata } from "next";
import { UploadStudio } from "./components/UploadStudio";

export const metadata: Metadata = {
  title: "Spatial Drop — Publish a model into the room",
  description:
    "Upload a GLB, publish a link, and let anyone place it in AR from a QR code.",
};

export default function Home() {
  return <UploadStudio />;
}
