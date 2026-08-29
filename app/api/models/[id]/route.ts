import { env } from "cloudflare:workers";

export async function GET(_request: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  if (!/^[a-f0-9]{12}$/.test(id)) return Response.json({ error: "Model not found." }, { status: 404 });
  const object = await env.MODELS.get(`metadata/${id}.json`);
  if (!object) return Response.json({ error: "Model not found." }, { status: 404 });
  return new Response(object.body, { headers: { "Content-Type": "application/json", "Cache-Control": "public, max-age=60" } });
}
