// Train on the object, not the room: after the camera solve the object's
// region is known, so every training photo is cropped to the box around that
// region and its camera's principal point moved to match. Fewer pixels per
// step and smaller targets in GPU memory; whatever background stays inside a
// crop is still learned at its real depth. (Masking the photos instead made
// the trainer explain the wall behind the object with floaters around it:
// 18.7 mm shape error against 3.4 mm, measured on the synthetic orbit.)

import type { subjectCylinder } from "./scan-mesh";

type Region = ReturnType<typeof subjectCylinder>;

/** A training frame as splat.js holds it (undistorted after the solve). */
export interface CropFrame {
  fw: number;
  fh: number;
  tw: number;
  th: number;
  rgb: Float32Array | null;
}

/** A solved camera at the solver's feature resolution, world-to-camera. */
export interface CropCamera {
  imgIdx: number;
  R: ArrayLike<number>;
  t: ArrayLike<number>;
  f: number;
  fy?: number;
  cx?: number;
  cy?: number;
}

/** Pixel box [x0, y0, x1, y1) of a width x height image where the region can
 *  appear, from rays through every `step`-th pixel; null when it never does. */
export function subjectBox(
  camera: Omit<CropCamera, "imgIdx">, region: Region, width: number, height: number,
  intrinsics: { fx: number; fy: number; cx: number; cy: number }, step = 4,
) {
  const { center, rotation: L, radius, bottom, top } = region;
  const R = camera.R, t = camera.t;
  // camera center and the camera-to-local rotation M = L · Rᵀ
  const C = [0, 1, 2].map((k) => -(R[k] * t[0] + R[3 + k] * t[1] + R[6 + k] * t[2]));
  const o = [0, 1, 2].map((r) => L[r * 3] * (C[0] - center[0]) + L[r * 3 + 1] * (C[1] - center[1]) + L[r * 3 + 2] * (C[2] - center[2]));
  const M = [0, 1, 2].flatMap((r) => [0, 1, 2].map((c) => L[r * 3] * R[c * 3] + L[r * 3 + 1] * R[c * 3 + 1] + L[r * 3 + 2] * R[c * 3 + 2]));
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let v = 0; v < height + step; v += step) {
    const py = Math.min(v, height - 1);
    const yc = (py + 0.5 - intrinsics.cy) / intrinsics.fy;
    for (let u = 0; u < width + step; u += step) {
      const px = Math.min(u, width - 1);
      const xc = (px + 0.5 - intrinsics.cx) / intrinsics.fx;
      const dx = M[0] * xc + M[1] * yc + M[2], dy = M[3] * xc + M[4] * yc + M[5], dz = M[6] * xc + M[7] * yc + M[8];
      // the ray's span inside the infinite cylinder x² + z² ≤ r² ...
      let near = 1e-6, far = Infinity;
      const a = dx * dx + dz * dz, b = 2 * (o[0] * dx + o[2] * dz), c = o[0] * o[0] + o[2] * o[2] - radius * radius;
      if (a < 1e-12) {
        if (c > 0) continue;
      } else {
        const disc = b * b - 4 * a * c;
        if (disc < 0) continue;
        const root = Math.sqrt(disc);
        near = Math.max(near, (-b - root) / (2 * a));
        far = Math.min(far, (-b + root) / (2 * a));
      }
      // ... and between its floor and top
      if (Math.abs(dy) < 1e-12) {
        if (o[1] < bottom || o[1] > top) continue;
      } else {
        const s0 = (bottom - o[1]) / dy, s1 = (top - o[1]) / dy;
        near = Math.max(near, Math.min(s0, s1));
        far = Math.min(far, Math.max(s0, s1));
      }
      if (far < near) continue;
      x0 = Math.min(x0, px); y0 = Math.min(y0, py);
      x1 = Math.max(x1, px + 1); y1 = Math.max(y1, py + 1);
    }
  }
  if (x1 < 0) return null;
  // the ray grid can step over the very edge: widen by a step, inside the image
  return [Math.max(0, x0 - step), Math.max(0, y0 - step), Math.min(width, x1 + step), Math.min(height, y1 + step)] as const;
}

/** Crop each solved frame to its subject box and shift its camera's principal
 *  point to match. Frames whose box is most of the image stay whole. Returns
 *  the fraction of training pixels kept. */
export function cropToSubject(frames: CropFrame[], cameras: CropCamera[], region: Region, options: { margin?: number; keepAbove?: number } = {}) {
  const margin = options.margin ?? 0.03;
  let before = 0, after = 0;
  for (const camera of cameras) {
    const frame = frames[camera.imgIdx];
    if (!frame.rgb) continue;
    // splat.js trains at s = tw / fw times the solver's feature scale; keep that ratio
    const s = frame.tw / frame.fw;
    const cx = (camera.cx ?? frame.fw / 2) * s, cy = (camera.cy ?? frame.fh / 2) * s;
    const intrinsics = { fx: camera.f * s, fy: (camera.fy ?? camera.f) * s, cx, cy };
    before += frame.tw * frame.th;
    const box = subjectBox(camera, region, frame.tw, frame.th, intrinsics);
    if (!box) { after += frame.tw * frame.th; continue; }
    const pad = margin * Math.max(frame.tw, frame.th);
    const x0 = Math.max(0, Math.floor(box[0] - pad)), y0 = Math.max(0, Math.floor(box[1] - pad));
    const x1 = Math.min(frame.tw, Math.ceil(box[2] + pad)), y1 = Math.min(frame.th, Math.ceil(box[3] + pad));
    const w = x1 - x0, h = y1 - y0;
    if (w * h > (options.keepAbove ?? 0.85) * frame.tw * frame.th) { after += frame.tw * frame.th; continue; }
    const rgb = new Float32Array(w * h * 3);
    for (let y = 0; y < h; y++) rgb.set(frame.rgb.subarray(((y0 + y) * frame.tw + x0) * 3, ((y0 + y) * frame.tw + x1) * 3), y * w * 3);
    frame.rgb = rgb;
    frame.tw = w;
    frame.th = h;
    frame.fw = w / s;
    frame.fh = h / s;
    camera.cx = (cx - x0) / s;
    camera.cy = (cy - y0) / s;
    after += w * h;
  }
  return before ? after / before : 1;
}
