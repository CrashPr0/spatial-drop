import type { Metadata } from "next";
import { notFound } from "next/navigation";
import Link from "next/link";
import { env } from "cloudflare:workers";
import { ModelViewer } from "../../components/ModelViewer";

type ModelMetadata = {
  id: string;
  title: string;
  placement: "floor" | "wall";
  resizable: boolean;
  filename: string;
  size: number;
  createdAt: string;
};

async function getModel(id: string): Promise<ModelMetadata | null> {
  const object = await env.MODELS.get(`metadata/${id}.json`);
  if (!object) return null;
  return object.json<ModelMetadata>();
}

export async function generateMetadata({ params }: { params: Promise<{ id: string }> }): Promise<Metadata> {
  const { id } = await params;
  const model = await getModel(id);
  return {
    title: model ? `${model.title} — Spatial Drop` : "Model not found — Spatial Drop",
    description: model ? `Place ${model.title} in your space using augmented reality.` : "This model could not be found.",
  };
}

export default async function ViewerPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const model = await getModel(id);
  if (!model) notFound();

  return (
    <main className="public-viewer">
      <header className="viewer-header">
        <Link className="brand" href="/"><span className="brand-glyph">S/</span><span>SPATIAL DROP</span></Link>
        <span>AR MODEL · {model.placement.toUpperCase()}</span>
      </header>
      <section className="public-stage">
        <ModelViewer
          src={`/api/models/${id}/file`}
          alt={`3D model of ${model.title}`}
          placement={model.placement}
          resizable={model.resizable}
          className="public-model-viewer"
        />
        <div className="model-caption">
          <p>READY TO PLACE</p>
          <h1>{model.title}</h1>
          <span>Move your phone to find a {model.placement === "floor" ? "flat surface" : "wall"}, then place the model.</span>
        </div>
      </section>
      <div className="viewer-help"><span>01 · Tap “View in your space”</span><span>02 · Scan the room</span><span>03 · Tap to place</span></div>
    </main>
  );
}
