import { env } from "cloudflare:workers";

const MAX_BYTES = 15 * 1024 * 1024;

export async function POST(request: Request) {
  const form = await request.formData();
  const file = form.get("model");
  if (!(file instanceof File)) return Response.json({ error: "No model was provided." }, { status: 400 });
  if (!file.name.toLowerCase().endsWith(".glb")) return Response.json({ error: "Only .glb files are supported." }, { status: 415 });
  if (file.size > MAX_BYTES) return Response.json({ error: "Models must be 15 MB or smaller." }, { status: 413 });

  const bytes = await file.arrayBuffer();
  const signature = new TextDecoder().decode(bytes.slice(0, 4));
  if (signature !== "glTF") return Response.json({ error: "This file is not a valid binary glTF model." }, { status: 422 });

  const id = crypto.randomUUID().replaceAll("-", "").slice(0, 12);
  const title = String(form.get("title") || "Untitled model").trim().slice(0, 80) || "Untitled model";
  const placement = form.get("placement") === "wall" ? "wall" : "floor";
  const resizable = form.get("resizable") !== "false";
  const metadata = { id, title, placement, resizable, filename: file.name, size: file.size, createdAt: new Date().toISOString() };

  await Promise.all([
    env.MODELS.put(`models/${id}.glb`, bytes, {
      httpMetadata: { contentType: "model/gltf-binary", cacheControl: "public, max-age=31536000, immutable" },
      customMetadata: { title },
    }),
    env.MODELS.put(`metadata/${id}.json`, JSON.stringify(metadata), {
      httpMetadata: { contentType: "application/json", cacheControl: "public, max-age=60" },
    }),
  ]);

  return Response.json({ id, title }, { status: 201 });
}
