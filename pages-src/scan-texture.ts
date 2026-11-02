// Paint the texture atlas from the original photos instead of the splat's
// colors. A splat stores one smooth color per Gaussian; the photos carry the
// full detail. Each texel takes a weighted blend of the views that see its
// surface point head-on and unoccluded, with the weight sharpened so the best
// view dominates and misregistration between views does not blur the result.

import type { ScanMesh, Vec3 } from "./scan-mesh";

export interface PhotoView {
  /** world-to-camera rotation (row-major) and translation, OpenCV axes */
  R: ArrayLike<number>;
  t: ArrayLike<number>;
  /** intrinsics at the solver's feature resolution */
  f: number;
  fy?: number;
  cx?: number;
  cy?: number;
  featureWidth: number;
  featureHeight: number;
  /** radial distortion of the original photo, on normalized coordinates */
  k1: number;
  k2: number;
  load: () => Promise<{ data: Uint8ClampedArray; width: number; height: number }>;
}

export interface TexelAtlas {
  size: number;
  pixels: Uint8ClampedArray;
  texelPoints: Float32Array;
  texelNormals: Float32Array;
  texelUsed: Uint8Array;
}

export interface ProjectionOptions {
  /** view weight = (cos(angle) / distance²) ^ sharpness: higher favors the single best view */
  sharpness?: number;
  /** depth buffer resolution as a fraction of the photo */
  depthScale?: number;
  onProgress?: (done: number, total: number) => void;
}

/** Rasterize the mesh's camera-space depth (nearest surface) at a low resolution. */
function depthBuffer(
  world: Float32Array, indices: Uint32Array, R: ArrayLike<number>, t: ArrayLike<number>,
  fx: number, fy: number, cx: number, cy: number, width: number, height: number,
) {
  const inverseDepth = new Float32Array(width * height); // 0 = nothing; larger = nearer
  const count = world.length / 3;
  const sx = new Float32Array(count), sy = new Float32Array(count), sw = new Float32Array(count);
  for (let v = 0; v < count; v++) {
    const X = world[v * 3], Y = world[v * 3 + 1], Z = world[v * 3 + 2];
    const x = R[0] * X + R[1] * Y + R[2] * Z + t[0];
    const y = R[3] * X + R[4] * Y + R[5] * Z + t[1];
    const z = R[6] * X + R[7] * Y + R[8] * Z + t[2];
    sw[v] = z > 1e-6 ? 1 / z : 0;
    sx[v] = fx * x * sw[v] + cx;
    sy[v] = fy * y * sw[v] + cy;
  }
  for (let f = 0; f < indices.length; f += 3) {
    const a = indices[f], b = indices[f + 1], c = indices[f + 2];
    if (!sw[a] || !sw[b] || !sw[c]) continue;
    const area = (sx[b] - sx[a]) * (sy[c] - sy[a]) - (sx[c] - sx[a]) * (sy[b] - sy[a]);
    if (Math.abs(area) < 1e-9) continue;
    const minX = Math.max(0, Math.floor(Math.min(sx[a], sx[b], sx[c])));
    const maxX = Math.min(width - 1, Math.ceil(Math.max(sx[a], sx[b], sx[c])));
    const minY = Math.max(0, Math.floor(Math.min(sy[a], sy[b], sy[c])));
    const maxY = Math.min(height - 1, Math.ceil(Math.max(sy[a], sy[b], sy[c])));
    for (let py = minY; py <= maxY; py++) {
      for (let px = minX; px <= maxX; px++) {
        const qx = px + 0.5, qy = py + 0.5;
        const wa = ((sx[b] - qx) * (sy[c] - qy) - (sx[c] - qx) * (sy[b] - qy)) / area;
        const wb = ((sx[c] - qx) * (sy[a] - qy) - (sx[a] - qx) * (sy[c] - qy)) / area;
        const wc = 1 - wa - wb;
        if (wa < 0 || wb < 0 || wc < 0) continue;
        // 1/z interpolates linearly in screen space
        const w = wa * sw[a] + wb * sw[b] + wc * sw[c];
        const i = py * width + px;
        if (w > inverseDepth[i]) inverseDepth[i] = w;
      }
    }
  }
  return inverseDepth;
}

/** Overwrite atlas texels with photo colors where any view sees them.
 *  Returns the fraction of used texels that got a photo color. */
export async function projectPhotos(atlas: TexelAtlas, mesh: ScanMesh, views: PhotoView[], options: ProjectionOptions = {}) {
  // held-out PSNR barely moves between 3 and 16 (24.6 vs 24.4 dB); 6 keeps small marks crisp
  const sharpness = options.sharpness ?? 6;
  const depthScale = options.depthScale ?? 0.25;
  const texels = atlas.size * atlas.size;

  // texel surface points and normals, and the mesh, in the cameras' world frame
  const used: number[] = [];
  for (let i = 0; i < texels; i++) if (atlas.texelUsed[i]) used.push(i);
  const points = new Float32Array(used.length * 3);
  const normals = new Float32Array(used.length * 3);
  used.forEach((texel, j) => {
    points.set(mesh.toWorld([atlas.texelPoints[texel * 3], atlas.texelPoints[texel * 3 + 1], atlas.texelPoints[texel * 3 + 2]]), j * 3);
    normals.set(mesh.directionToWorld([atlas.texelNormals[texel * 3], atlas.texelNormals[texel * 3 + 1], atlas.texelNormals[texel * 3 + 2]]), j * 3);
  });
  const world = new Float32Array(mesh.positions.length);
  for (let v = 0; v < mesh.positions.length; v += 3) {
    world.set(mesh.toWorld([mesh.positions[v], mesh.positions[v + 1], mesh.positions[v + 2]]), v);
  }

  // distances relative to the typical camera distance: the solve has no units
  const centroid: Vec3 = [0, 0, 0];
  for (let v = 0; v < world.length; v += 3) for (let k = 0; k < 3; k++) centroid[k] += world[v + k] / (world.length / 3);
  const reach = views.map(({ R, t }) => Math.hypot(
    -(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]) - centroid[0],
    -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]) - centroid[1],
    -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2]) - centroid[2],
  )).sort((a, b) => a - b)[views.length >> 1] || 1;

  const sum = new Float32Array(used.length * 3);
  const weight = new Float32Array(used.length);
  let viewIndex = 0;
  for (const view of views) {
    options.onProgress?.(viewIndex++, views.length);
    let image: Awaited<ReturnType<PhotoView["load"]>>;
    try {
      image = await view.load();
    } catch {
      continue;
    }
    const scale = image.width / view.featureWidth;
    // a photo decoded in another orientation than the solver saw is useless here
    if (Math.abs(image.height / view.featureHeight - scale) > 0.02 * scale) continue;
    const fx = view.f * scale, fy = (view.fy ?? view.f) * scale;
    const cx = (view.cx ?? view.featureWidth / 2) * scale, cy = (view.cy ?? view.featureHeight / 2) * scale;
    const { R, t } = view;
    const camera: Vec3 = [
      -(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]),
      -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]),
      -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2]),
    ];
    const dw = Math.max(16, Math.round(image.width * depthScale)), dh = Math.max(16, Math.round(image.height * depthScale));
    const ds = dw / image.width;
    const depth = depthBuffer(world, mesh.indices, R, t, fx * ds, fy * ds, cx * ds, cy * ds, dw, dh);
    const margin = 0.03 * Math.min(image.width, image.height);

    for (let j = 0; j < used.length; j++) {
      const X = points[j * 3], Y = points[j * 3 + 1], Z = points[j * 3 + 2];
      const z = R[6] * X + R[7] * Y + R[8] * Z + t[2];
      if (z <= 1e-6) continue;
      const toCamera: Vec3 = [camera[0] - X, camera[1] - Y, camera[2] - Z];
      const distance = Math.hypot(toCamera[0], toCamera[1], toCamera[2]);
      const facing = (normals[j * 3] * toCamera[0] + normals[j * 3 + 1] * toCamera[1] + normals[j * 3 + 2] * toCamera[2]) / distance;
      if (facing < 0.2) continue;
      const xn = (R[0] * X + R[1] * Y + R[2] * Z + t[0]) / z;
      const yn = (R[3] * X + R[4] * Y + R[5] * Z + t[1]) / z;
      // occlusion: something nearer than this point covers its depth pixel
      const px = Math.floor((fx * xn + cx) * ds), py = Math.floor((fy * yn + cy) * ds);
      if (px < 0 || py < 0 || px >= dw || py >= dh) continue;
      const nearest = depth[py * dw + px];
      if (nearest > 0 && 1 / nearest < z * 0.97) continue;
      // the photo itself is distorted: re-apply the lens model
      const r2 = xn * xn + yn * yn;
      const D = 1 + view.k1 * r2 + view.k2 * r2 * r2;
      const u = fx * xn * D + cx - 0.5, v = fy * yn * D + cy - 0.5;
      if (u < 0 || v < 0 || u > image.width - 1.001 || v > image.height - 1.001) continue;
      const edge = Math.min(1, Math.min(u, v, image.width - 1 - u, image.height - 1 - v) / margin);
      const relative = distance / reach;
      const w = (facing / (relative * relative)) ** sharpness * edge;
      if (!(w > 0)) continue;
      const x0 = u | 0, y0 = v | 0, ax = u - x0, ay = v - y0;
      const i00 = (y0 * image.width + x0) * 4, i10 = i00 + 4, i01 = i00 + image.width * 4, i11 = i01 + 4;
      for (let c = 0; c < 3; c++) {
        const value = (image.data[i00 + c] * (1 - ax) + image.data[i10 + c] * ax) * (1 - ay)
          + (image.data[i01 + c] * (1 - ax) + image.data[i11 + c] * ax) * ay;
        sum[j * 3 + c] += w * value;
      }
      weight[j] += w;
    }
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  options.onProgress?.(views.length, views.length);

  let painted = 0;
  used.forEach((texel, j) => {
    if (!(weight[j] > 0)) return;
    painted++;
    for (let c = 0; c < 3; c++) atlas.pixels[texel * 4 + c] = Math.round(sum[j * 3 + c] / weight[j]);
  });
  return used.length ? painted / used.length : 0;
}
