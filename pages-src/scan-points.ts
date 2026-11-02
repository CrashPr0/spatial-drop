// Without WebGPU there is no splat training, but the camera solve still
// leaves a sparse cloud of 3D points on the object. Each point becomes a
// round, nearly opaque splat sized to the gap to its neighbors, so the
// mesher's density shell and depth carving can run on it unchanged.

import type { SplatCloud } from "./scan-mesh";

const STRIDE = 16;
const SH_C0 = 0.28209479177387814;

export interface SolvedPoint {
  X: ArrayLike<number>;
  /** 0–1 */
  rgb: ArrayLike<number>;
}

/** Distance from each point to its k-th nearest neighbor, through a uniform grid. */
export function neighborSpacing(points: ArrayLike<number>[], k = 4) {
  const n = points.length;
  const spacing = new Float32Array(n);
  if (n <= k) return spacing.fill(0.01);
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (const p of points) for (let a = 0; a < 3; a++) { min[a] = Math.min(min[a], p[a]); max[a] = Math.max(max[a], p[a]); }
  // about two points per cell if they filled the box
  const volume = Math.max(1e-12, (max[0] - min[0]) * (max[1] - min[1]) * (max[2] - min[2]));
  const cell = Math.max(1e-6, Math.cbrt((2 * volume) / n));
  const key = (x: number, y: number, z: number) => `${x},${y},${z}`;
  const cellOf = (p: ArrayLike<number>) => [0, 1, 2].map((a) => Math.floor((p[a] - min[a]) / cell));
  const grid = new Map<string, number[]>();
  points.forEach((p, i) => {
    const [x, y, z] = cellOf(p);
    const id = key(x, y, z);
    const list = grid.get(id);
    if (list) list.push(i); else grid.set(id, [i]);
  });
  const cells = [0, 1, 2].map((a) => Math.ceil((max[a] - min[a]) / cell) + 1);
  const best: number[] = [];
  for (let i = 0; i < n; i++) {
    const p = points[i];
    const [cx, cy, cz] = cellOf(p);
    // widen the search shell until k neighbors are found and none closer can hide outside it
    for (let ring = 1; ; ring++) {
      best.length = 0;
      for (let x = cx - ring; x <= cx + ring; x++) for (let y = cy - ring; y <= cy + ring; y++) for (let z = cz - ring; z <= cz + ring; z++) {
        const list = grid.get(key(x, y, z));
        if (!list) continue;
        for (const j of list) {
          if (j === i) continue;
          const q = points[j];
          best.push(Math.hypot(q[0] - p[0], q[1] - p[1], q[2] - p[2]));
        }
      }
      best.sort((a, b) => a - b);
      if ((best.length >= k && best[k - 1] <= ring * cell) || ring > Math.max(...cells)) {
        spacing[i] = best[Math.min(k, best.length) - 1] ?? cell;
        break;
      }
    }
  }
  return spacing;
}

/**
 * A splat cloud standing in for a trained one, from the solve's points.
 * Measured on the synthetic orbit (1,504 points), against 2.7–3.1 mm and
 * 25 dB for a trained splat: k 3, size 0.3 gives 11.8 mm and 20.6 dB held
 * out; bigger splats bulge (k 4, size 0.7: 18 mm).
 */
export function pointSplats(points: SolvedPoint[], { k = 3, size = 0.3, opacity = 0.9 } = {}): SplatCloud {
  const spacing = neighborSpacing(points.map((p) => p.X), k);
  // stray points far from the rest would become huge blobs: cap at the 90th percentile
  const sorted = Array.from(spacing).sort((a, b) => a - b);
  const cap = sorted[Math.floor(sorted.length * 0.9)] ?? 0.01;
  const data = new Float32Array(points.length * STRIDE);
  const logitOpacity = Math.log(opacity / (1 - opacity));
  points.forEach((point, i) => {
    const o = i * STRIDE;
    const scale = Math.log(Math.max(1e-5, size * Math.min(spacing[i], cap)));
    data[o] = point.X[0]; data[o + 1] = point.X[1]; data[o + 2] = point.X[2];
    data[o + 3] = data[o + 4] = data[o + 5] = scale;
    data[o + 6] = 1;
    for (let c = 0; c < 3; c++) data[o + 10 + c] = (Math.min(1, Math.max(0, point.rgb[c])) - 0.5) / SH_C0;
    data[o + 13] = logitOpacity;
  });
  return { data, n: points.length, dc: "sh" };
}
