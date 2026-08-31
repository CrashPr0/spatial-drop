import type { Metadata } from "next";
import { UploadStudio } from "../components/UploadStudio";

export const metadata: Metadata = {
  title: "Spatial Drop — Publisher Prototype",
  description: "Upload a GLB, publish a link, and create an AR-ready QR code.",
};

export default function PublishPage() {
  return <UploadStudio />;
}
