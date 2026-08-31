import type { Metadata } from "next";
import { Geist_Mono, Nunito_Sans } from "next/font/google";
import { headers } from "next/headers";
import "./globals.css";

const nunitoSans = Nunito_Sans({ variable: "--font-nunito-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export async function generateMetadata(): Promise<Metadata> {
  const requestHeaders = await headers();
  const host = requestHeaders.get("x-forwarded-host") ?? requestHeaders.get("host") ?? "localhost:3000";
  const protocol = requestHeaders.get("x-forwarded-proto") ?? (host.startsWith("localhost") ? "http" : "https");
  return {
    metadataBase: new URL(`${protocol}://${host}`),
    title: { default: "Spatial Drop", template: "%s" },
    description: "An SJSU-inspired spatial computing prototype for previewing 3D models in the browser and in AR.",
    openGraph: {
      title: "Spatial Drop — Bring ideas into the room",
      description: "Preview a 3D model, create a QR link, and place it in your space.",
      type: "website",
      images: [{ url: "/og-sjsu.png", width: 1536, height: 1024, alt: "Spatial Drop — bring ideas into the room" }],
    },
    twitter: {
      card: "summary_large_image",
      title: "Spatial Drop — Bring ideas into the room",
      description: "Preview a 3D model, create a QR link, and place it in your space.",
      images: ["/og-sjsu.png"],
    },
  };
}

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className={`${nunitoSans.variable} ${geistMono.variable}`}>{children}</body>
    </html>
  );
}
