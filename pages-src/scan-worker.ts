// Meshing and texturing off the main thread: carving, decimation and photo
// projection take seconds, and in a hidden tab the main thread's timers are
// throttled to a crawl. One job per worker; the page terminates it to cancel.

/// <reference lib="webworker" />

import { bakeTextureAtlas, buildScanMesh, type MeshOptions, type ScanCamera, type SplatCloud } from "./scan-mesh";
import { chooseShape, mightBeShape, snapToShape } from "./scan-shapes";
import { measureShine } from "./scan-shine";
import { projectPhotos, type PhotoView } from "./scan-texture";

// photos are decoded at most this large for texturing
const TEXTURE_PHOTO_MAX = 1600;
// and this large for judging the surface's shine (1.9 s for 57 photos on a laptop)
const SHINE_PHOTO_MAX = 800;

/** Camera data the texture projector needs, without the photo itself. */
export interface ViewMeta {
  name: string;
  R: number[];
  t: number[];
  f: number;
  fy?: number;
  cx?: number;
  cy?: number;
  featureWidth: number;
  featureHeight: number;
  k1: number;
  k2: number;
}

export interface MeshJob {
  splat: SplatCloud;
  cameras: ScanCamera[];
  viewMetas: ViewMeta[];
  photos: File[];
  options: Omit<MeshOptions, "views">;
  /** "auto": a scan that is a can, a bottle or a box becomes that clean shape */
  shapes: "auto" | "scanned";
  textureSize: number;
}

export type MeshWorkerMessage =
  | { type: "progress"; stage: "mesh" | "shape" | "texture"; done: number; total: number }
  | {
    type: "done";
    mesh: { positions: Float32Array; normals: Float32Array; indices: Uint32Array; stats: ReturnType<typeof buildScanMesh>["stats"] };
    atlas: Omit<ReturnType<typeof bakeTextureAtlas>, "texelPoints" | "texelNormals" | "texelUsed">;
    /** milliseconds per step, for profiling */
    timings: { mesh: number; shape: number; atlas: number; photos: number };
  }
  | { type: "error"; message: string };

async function decodePhoto(file: Blob, largest = TEXTURE_PHOTO_MAX) {
  const bitmap = await createImageBitmap(file);
  const scale = Math.min(1, largest / Math.max(bitmap.width, bitmap.height));
  const width = Math.round(bitmap.width * scale), height = Math.round(bitmap.height * scale);
  const canvas = new OffscreenCanvas(width, height);
  try {
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) throw new Error("Photo decoding needs a 2D canvas context.");
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = "high";
    context.drawImage(bitmap, 0, 0, width, height);
    return context.getImageData(0, 0, width, height);
  } finally {
    bitmap.close();
    // Release backing stores promptly on memory-constrained mobile browsers.
    canvas.width = canvas.height = 0;
  }
}

/** Pair each solved camera with its photo by file name (frames that failed to decode are skipped). */
function photoViews(metas: ViewMeta[], files: File[], largest?: number): PhotoView[] {
  const byName = new Map<string, File[]>();
  for (const file of files) byName.set(file.name, [...(byName.get(file.name) ?? []), file]);
  return metas.flatMap((meta) => {
    const file = byName.get(meta.name)?.shift();
    return file ? [{ ...meta, load: () => decodePhoto(file, largest) }] : [];
  });
}

const post = (message: MeshWorkerMessage, transfer: Transferable[] = []) => self.postMessage(message, transfer);

self.onmessage = async ({ data: job }: MessageEvent<MeshJob>) => {
  try {
    const started = performance.now();
    post({ type: "progress", stage: "mesh", done: 0, total: 1 });
    let mesh = buildScanMesh(job.splat, job.cameras, {
      ...job.options,
      // carve from what the cameras saw, so a gappy splat shell cannot cave the model in
      views: job.viewMetas.map((meta) => ({
        R: meta.R, t: meta.t, fx: meta.f, fy: meta.fy ?? meta.f,
        cx: meta.cx ?? meta.featureWidth / 2, cy: meta.cy ?? meta.featureHeight / 2,
        width: meta.featureWidth, height: meta.featureHeight,
      })),
    });
    const meshed = performance.now();
    if (job.shapes === "auto") {
      // only read the photos for shine when a shape could fit at all
      const sample = { positions: mesh.positions, normals: mesh.normals, bottom: 0 };
      const views = mightBeShape(sample) ? photoViews(job.viewMetas, job.photos, SHINE_PHOTO_MAX) : [];
      if (views.length) {
        post({ type: "progress", stage: "shape", done: 0, total: 1 });
        const { shine } = await measureShine(mesh, views);
        const fit = Number.isFinite(shine) ? chooseShape(sample, shine) : null;
        if (fit) mesh = snapToShape(mesh, fit, job.options.targetTriangles);
      }
    }
    const shaped = performance.now();
    const atlas = bakeTextureAtlas(mesh, { size: job.textureSize });
    const baked = performance.now();
    const views = photoViews(job.viewMetas, job.photos);
    if (views.length) {
      // the photos hold far more detail than the splat's colors
      await projectPhotos(atlas, mesh, views, {
        onProgress: (done, total) => post({ type: "progress", stage: "texture", done, total }),
      });
    }
    // the per-texel surface data stays here; the page only needs the painted atlas
    const painted = { size: atlas.size, cell: atlas.cell, pixels: atlas.pixels, indices: atlas.indices, positions: atlas.positions, normals: atlas.normals, uvs: atlas.uvs };
    post(
      {
        type: "done", mesh: { positions: mesh.positions, normals: mesh.normals, indices: mesh.indices, stats: mesh.stats }, atlas: painted,
        timings: { mesh: Math.round(meshed - started), shape: Math.round(shaped - meshed), atlas: Math.round(baked - shaped), photos: Math.round(performance.now() - baked) },
      },
      [mesh.positions.buffer, mesh.normals.buffer, mesh.indices.buffer, painted.pixels.buffer, painted.indices.buffer,
        painted.positions.buffer, painted.normals.buffer, painted.uvs.buffer],
    );
  } catch (reason) {
    post({ type: "error", message: reason instanceof Error ? reason.message : "The mesh could not be built." });
  }
};
