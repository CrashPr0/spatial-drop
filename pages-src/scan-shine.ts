// How shiny the scanned surface is, from the photos. A matte point looks the
// same from every side; a shiny one shows whatever it reflects, so its color
// swings from photo to photo as the camera moves. A model that sits slightly
// off the real surface also changes color between photos, but only where
// the photo is busy around the point (lettering on a label), so the swing
// is measured against the image's own local variation there.

import type { ScanMesh, Vec3 } from "./scan-mesh";
import { depthBuffer, type PhotoView } from "./scan-texture";

export interface ShineReport {
  /** median over the surface of the swing between photos relative to local image variation; matte stays near 1 */
  shine: number;
  /** surface points seen by enough photos to judge */
  judged: number;
}

const luminance = (data: Uint8ClampedArray, i: number) => 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];

function medianOf(values: number[]) {
  const sorted = values.slice().sort((a, b) => a - b);
  return sorted.length ? sorted[sorted.length >> 1] : NaN;
}

/** Judge every `stride`-th vertex of the mesh from every photo that sees it head-on. */
export async function measureShine(
  mesh: Pick<ScanMesh, "positions" | "normals" | "indices" | "toWorld" | "directionToWorld">,
  views: PhotoView[], { stride = 4, minViews = 5 } = {},
): Promise<ShineReport> {
  const count = mesh.positions.length / 3;
  const world = new Float32Array(mesh.positions.length);
  for (let v = 0; v < count; v++) world.set(mesh.toWorld([mesh.positions[v * 3], mesh.positions[v * 3 + 1], mesh.positions[v * 3 + 2]]), v * 3);
  const sampled: number[] = [];
  for (let v = 0; v < count; v += stride) sampled.push(v);
  const normals = sampled.map((v) => mesh.directionToWorld([mesh.normals[v * 3], mesh.normals[v * 3 + 1], mesh.normals[v * 3 + 2]]));
  // every sighting: which point, which photo, the luminance there and how busy the photo is around it
  const point: number[] = [], photo: number[] = [], level: number[] = [], busy: number[] = [];
  let photoIndex = -1;

  for (const view of views) {
    photoIndex++;
    let image: Awaited<ReturnType<PhotoView["load"]>>;
    try {
      image = await view.load();
    } catch {
      continue;
    }
    const scale = image.width / view.featureWidth;
    if (Math.abs(image.height / view.featureHeight - scale) > 0.02 * scale) continue;
    const fx = view.f * scale, fy = (view.fy ?? view.f) * scale;
    const cx = (view.cx ?? view.featureWidth / 2) * scale, cy = (view.cy ?? view.featureHeight / 2) * scale;
    const { R, t } = view;
    const camera: Vec3 = [
      -(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]),
      -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]),
      -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2]),
    ];
    const dw = Math.max(16, Math.round(image.width / 4)), dh = Math.max(16, Math.round(image.height / 4));
    const ds = dw / image.width;
    const depth = depthBuffer(world, mesh.indices, R, t, fx * ds, fy * ds, cx * ds, cy * ds, dw, dh);
    // the window for local variation: about a hundredth of the photo
    const half = Math.max(2, Math.round(0.006 * Math.max(image.width, image.height)));
    sampled.forEach((v, j) => {
      const X = world[v * 3], Y = world[v * 3 + 1], Z = world[v * 3 + 2];
      const z = R[6] * X + R[7] * Y + R[8] * Z + t[2];
      if (z <= 1e-6) return;
      const toCamera: Vec3 = [camera[0] - X, camera[1] - Y, camera[2] - Z];
      const facing = (normals[j][0] * toCamera[0] + normals[j][1] * toCamera[1] + normals[j][2] * toCamera[2]) / Math.hypot(...toCamera);
      if (facing < 0.5) return;
      const xn = (R[0] * X + R[1] * Y + R[2] * Z + t[0]) / z;
      const yn = (R[3] * X + R[4] * Y + R[5] * Z + t[1]) / z;
      const px = Math.floor((fx * xn + cx) * ds), py = Math.floor((fy * yn + cy) * ds);
      if (px < 0 || py < 0 || px >= dw || py >= dh) return;
      const nearest = depth[py * dw + px];
      if (nearest > 0 && 1 / nearest < z * 0.97) return;
      const r2 = xn * xn + yn * yn;
      const D = 1 + view.k1 * r2 + view.k2 * r2 * r2;
      const u = Math.round(fx * xn * D + cx - 0.5), w = Math.round(fy * yn * D + cy - 0.5);
      if (u < half || w < half || u >= image.width - half || w >= image.height - half) return;
      let sum = 0, square = 0, n = 0;
      for (let dy = -half; dy <= half; dy += 2) for (let dx = -half; dx <= half; dx += 2) {
        const L = luminance(image.data, ((w + dy) * image.width + u + dx) * 4);
        sum += L; square += L * L; n++;
      }
      const mean = sum / n;
      point.push(j); photo.push(photoIndex); level.push(mean);
      busy.push(Math.sqrt(Math.max(0, square / n - mean * mean)));
    });
    await new Promise((resolve) => setTimeout(resolve, 0));
  }

  const byPoint = (values: number[]) => {
    const lists: number[][] = sampled.map(() => []);
    values.forEach((value, k) => lists[point[k]].push(value));
    return lists;
  };
  // a phone sets its exposure per photo: the same matte surface comes out
  // brighter in one and darker in the next. Each photo's gain is its median
  // ratio to the points' typical levels, divided out before judging
  const typical = byPoint(level).map(medianOf);
  const ratiosPerPhoto: number[][] = views.map(() => []);
  level.forEach((value, k) => {
    if (typical[point[k]] > 8) ratiosPerPhoto[photo[k]].push(value / typical[point[k]]);
  });
  const gain = ratiosPerPhoto.map((list) => (list.length >= 20 ? medianOf(list) : 1));
  const levels = byPoint(level.map((value, k) => value / gain[photo[k]]));
  const busyness = byPoint(busy.map((value, k) => value / gain[photo[k]]));
  const ratios: number[] = [];
  levels.forEach((values, j) => {
    if (values.length < minViews) return;
    const middle = medianOf(values);
    const swing = medianOf(values.map((value) => Math.abs(value - middle)));
    // a few gray levels of floor: flat, dark patches would otherwise divide by nothing
    ratios.push(swing / (medianOf(busyness[j]) + 3));
  });
  return { shine: ratios.length ? medianOf(ratios) : NaN, judged: ratios.length };
}
