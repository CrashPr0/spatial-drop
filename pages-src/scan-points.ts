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

/** Distance to the k-th nearest neighbor, using a balanced spatial tree.
 * Unlike a volume-based grid, this also handles flat scans and isolated points.
 */
export function neighborSpacing(points: ArrayLike<number>[], k = 4) {
  if (!Number.isInteger(k) || k < 1) throw new RangeError("k must be a positive integer");
  const n = points.length;
  const spacing = new Float32Array(n);
  if (n <= k) return spacing.fill(0.01);
  interface Node { index: number; axis: number; left: Node | null; right: Node | null }
  const build = (indices: number[]): Node | null => {
    if (!indices.length) return null;
    const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
    for (const i of indices) for (let a = 0; a < 3; a++) {
      min[a] = Math.min(min[a], points[i][a]); max[a] = Math.max(max[a], points[i][a]);
    }
    let axis = 0;
    for (let a = 1; a < 3; a++) if (max[a] - min[a] > max[axis] - min[axis]) axis = a;
    indices.sort((a, b) => points[a][axis] - points[b][axis]);
    const mid = indices.length >> 1;
    return { index: indices[mid], axis, left: build(indices.slice(0, mid)), right: build(indices.slice(mid + 1)) };
  };
  const root = build(Array.from({ length: n }, (_, i) => i));
  for (let i = 0; i < n; i++) {
    const p = points[i], best: number[] = [];
    const visit = (node: Node | null): void => {
      if (!node) return;
      const q = points[node.index];
      const delta = p[node.axis] - q[node.axis];
      if (node.index !== i) {
        const distance = (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2;
        if (best.length < k || distance < best[best.length - 1]) {
          let at = best.length;
          while (at > 0 && best[at - 1] > distance) at--;
          best.splice(at, 0, distance);
          if (best.length > k) best.pop();
        }
      }
      visit(delta <= 0 ? node.left : node.right);
      if (best.length < k || delta * delta < best[k - 1]) visit(delta <= 0 ? node.right : node.left);
    };
    visit(root);
    spacing[i] = Math.sqrt(best[k - 1]);
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
