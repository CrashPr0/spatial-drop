import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { headers } from "next/headers";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export async function generateMetadata(): Promise<Metadata> {
  const requestHeaders = await headers();
  const host = requestHeaders.get("x-forwarded-host") ?? requestHeaders.get("host") ?? "localhost:3000";
  const protocol = requestHeaders.get("x-forwarded-proto") ?? (host.startsWith("localhost") ? "http" : "https");
  return {
    metadataBase: new URL(`${protocol}://${host}`),
    title: { default: "Spatial Drop", template: "%s" },
    description: "Upload a 3D model, share a QR code, and place it in augmented reality.",
    openGraph: {
      title: "Spatial Drop — Put your model in the room",
      description: "One GLB. One QR code. No app download.",
      type: "website",
      images: [{ url: "/og.png", width: 1728, height: 910, alt: "Spatial Drop — put your model in the room" }],
    },
    twitter: {
      card: "summary_large_image",
      title: "Spatial Drop — Put your model in the room",
      description: "One GLB. One QR code. No app download.",
      images: ["/og.png"],
    },
  };
}

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable}`}>{children}</body>
    </html>
  );
}
