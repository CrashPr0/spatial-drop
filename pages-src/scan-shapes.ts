// Simple shapes, fitted to a carved mesh. Many scanned things are a solid of
// revolution standing upright (cans, bottles, cups, vases) or a box, and a
// scan of them comes out lumpy where the surface misled the trainer: shiny
// aluminium reflects the room, so the splat puts surface where the
// reflections are. When one of these shapes explains the surface, the clean
// shape is the better model; the photos still paint it.
//
// How sure the fit has to be depends on the surface. A matte scan is
// trustworthy, so it only becomes a shape when it nearly is one already. A
// shiny scan is dented wherever reflections sat behind the surface, so its
// outer envelope is fitted instead, and it becomes a round shape unless some
// side sticks out past that (a handle, a spout) or a box fits it better.
//
// Everything works in the mesher's leveled frame: +Y up, the object standing
// on the plane y = bottom.

import { computeNormals, decimate, type ScanMesh, type Vec3 } from "./scan-mesh";

export interface RevolveFit {
  kind: "revolve";
  /** axis position in x, z */
  center: [number, number];
  bottom: number;
  top: number;
  /** radius at the middle of each of `profile.length` equal height bands */
  profile: Float32Array;
  /** share of the side surface within tolerance of the shape */
  inliers: number;
  /** median distance from the side surface to the shape, relative to the mean radius */
  residual: number;
  /**
   * The most the surface sticks out past the shape on any one side (median
   * over that side, relative to the radius). A handle or a spout does; the
   * dents a reflection leaves do not.
   */
  protrusion: number;
}

export interface BoxFit {
  kind: "box";
  center: [number, number];
  /** turn about +Y of the box's first axis, radians */
  angle: number;
  /** half extents along the box's two horizontal axes */
  half: [number, number];
  bottom: number;
  top: number;
  inliers: number;
  /** median distance from the side surface to the shape, relative to the half of the box's smaller side */
  residual: number;
}

export type ShapeFit = RevolveFit | BoxFit;

export interface SurfaceSample {
  positions: Float32Array;
  normals: Float32Array;
  bottom: number;
}

const BANDS = 32;
/** sectors around the axis for judging one side at a time */
const SIDES = 16;

function median(values: number[]) {
  if (!values.length) return NaN;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[sorted.length >> 1];
}

function quantile(values: number[], q: number) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.floor(q * (sorted.length - 1))))];
}

/** at most about this many side points are fitted: the box search sorts them 90 times */
const MAX_SIDE_POINTS = 4000;

/** Side points (normals mostly horizontal) and the object's top, from a closed mesh. */
function sideOf({ positions, normals, bottom }: SurfaceSample) {
  const side: Vec3[] = [];
  const heights: number[] = [];
  const count = positions.length / 3;
  const stride = Math.max(1, Math.floor(count / (2 * MAX_SIDE_POINTS)));
  for (let v = 0; v < count; v += stride) {
    const y = positions[v * 3 + 1];
    heights.push(y);
    if (Math.abs(normals[v * 3 + 1]) < 0.5) side.push([positions[v * 3], y, positions[v * 3 + 2]]);
  }
  const top = quantile(heights, 0.995);
  return { side: side.filter((p) => p[1] > bottom && p[1] < top), top };
}

/**
 * The upright solid of revolution closest to the side surface: its axis by
 * Gauss-Newton on the spread of radii within each height band, then the
 * radius per band. Lumps from either side cancel in the band medians.
 */
export function fitRevolve(sample: SurfaceSample, { tolerance = 0.08, envelope = 0.5 } = {}): RevolveFit | null {
  const { side, top } = sideOf(sample);
  const { bottom } = sample;
  if (side.length < 60 || top <= bottom) return null;
  const band = (y: number) => Math.min(BANDS - 1, Math.max(0, Math.floor((y - bottom) / (top - bottom) * BANDS)));
  let cx = median(side.map((p) => p[0])), cz = median(side.map((p) => p[2]));
  const radii = new Float32Array(BANDS);
  const bandRadii = (q: number) => {
    const lists: number[][] = Array.from({ length: BANDS }, () => []);
    for (const p of side) lists[band(p[1])].push(Math.hypot(p[0] - cx, p[2] - cz));
    for (let b = 0; b < BANDS; b++) radii[b] = lists[b].length ? quantile(lists[b], q) : NaN;
  };
  for (let iteration = 0; iteration < 12; iteration++) {
    bandRadii(0.5);
    // minimize Σ (r − R_band)² over the center: 2x2 normal equations
    let a00 = 0, a01 = 0, a11 = 0, b0 = 0, b1 = 0;
    for (const p of side) {
      const dx = p[0] - cx, dz = p[2] - cz;
      const r = Math.hypot(dx, dz);
      const R = radii[band(p[1])];
      if (r < 1e-9 || !Number.isFinite(R)) continue;
      const e = r - R;
      // Huber weight: lumps far off the shape pull less
      const w = Math.abs(e) < 0.1 * R ? 1 : (0.1 * R) / Math.abs(e);
      const jx = -dx / r, jz = -dz / r;
      a00 += w * jx * jx; a01 += w * jx * jz; a11 += w * jz * jz;
      b0 -= w * jx * e; b1 -= w * jz * e;
    }
    const det = a00 * a11 - a01 * a01;
    if (Math.abs(det) < 1e-12) break;
    const sx = (b0 * a11 - b1 * a01) / det, sz = (a00 * b1 - a01 * b0) / det;
    cx += sx; cz += sz;
    if (Math.hypot(sx, sz) < 1e-6 * (top - bottom)) break;
  }
  // the axis from the median, the radius from `envelope`: a shiny surface's
  // scan is dented where reflections sat behind it, and its outer part is the
  // real surface (a can, 3.3 cm: median 2.9 cm, 75th percentile 3.28 cm)
  bandRadii(envelope);
  // fill empty bands, then smooth the profile: [1 2 1] after a median of three
  for (let b = 0; b < BANDS; b++) {
    if (Number.isFinite(radii[b])) continue;
    let lo = b - 1, hi = b + 1;
    while (lo >= 0 && !Number.isFinite(radii[lo])) lo--;
    while (hi < BANDS && !Number.isFinite(radii[hi])) hi++;
    radii[b] = lo < 0 ? radii[hi] : hi >= BANDS ? radii[lo] : radii[lo] + (radii[hi] - radii[lo]) * (b - lo) / (hi - lo);
  }
  if (!radii.every(Number.isFinite)) return null;
  const medians = radii.map((_, b) => median([radii[Math.max(0, b - 1)], radii[b], radii[Math.min(BANDS - 1, b + 1)]]));
  let profile = medians.map((_, b) => 0.25 * medians[Math.max(0, b - 1)] + 0.5 * medians[b] + 0.25 * medians[Math.min(BANDS - 1, b + 1)]);
  const mean = profile.reduce((sum, r) => sum + r, 0) / BANDS;
  const errorsFor = (radiusAt: (b: number) => number) => side.map((p) => (Math.hypot(p[0] - cx, p[2] - cz) - radiusAt(band(p[1]))) / mean);
  // a straight cylinder unless the profile explains the surface clearly
  // better: dents fall unevenly over the height, and a can given a waist by
  // them looks worse than a can. Judged on the mean error, not the median:
  // a bottle's neck is a small share of its surface, not the median: a bottle's neck is a small share of its surface)
  const straight = quantile(Array.from(profile), 0.5);
  const meanError = (errors: number[]) => errors.reduce((sum, e) => sum + Math.abs(e), 0) / errors.length;
  if (meanError(errorsFor(() => straight)) <= 1.2 * meanError(errorsFor((b) => profile[b])) + 0.005) profile = profile.fill(straight);
  const signed = errorsFor((b) => profile[b]);
  const errors = signed.map(Math.abs);
  const sides: number[][] = Array.from({ length: SIDES }, () => []);
  side.forEach((p, i) => sides[Math.floor((Math.atan2(p[2] - cz, p[0] - cx) / (2 * Math.PI) + 0.5) * SIDES) % SIDES].push(signed[i]));
  return {
    kind: "revolve", center: [cx, cz], bottom, top, profile,
    inliers: errors.filter((e) => e < tolerance).length / errors.length,
    residual: median(errors),
    protrusion: Math.max(...sides.filter((list) => list.length >= 10).map(median)),
  };
}

/** The upright box closest to the side surface: the smallest robust footprint over turns of 1°. */
export function fitBox(sample: SurfaceSample, { tolerance = 0.08 } = {}): BoxFit | null {
  const { side, top } = sideOf(sample);
  const { bottom } = sample;
  if (side.length < 60 || top <= bottom) return null;
  let best: Omit<BoxFit, "inliers" | "residual" | "kind" | "bottom" | "top"> | null = null;
  let bestArea = Infinity;
  for (let degrees = 0; degrees < 90; degrees++) {
    const angle = degrees * Math.PI / 180, c = Math.cos(angle), s = Math.sin(angle);
    const us = side.map((p) => c * p[0] + s * p[2]), vs = side.map((p) => -s * p[0] + c * p[2]);
    const u0 = quantile(us, 0.01), u1 = quantile(us, 0.99), v0 = quantile(vs, 0.01), v1 = quantile(vs, 0.99);
    const area = (u1 - u0) * (v1 - v0);
    if (area >= bestArea) continue;
    bestArea = area;
    const mu = (u0 + u1) / 2, mv = (v0 + v1) / 2;
    best = { center: [c * mu - s * mv, s * mu + c * mv], angle, half: [(u1 - u0) / 2, (v1 - v0) / 2] };
  }
  if (!best) return null;
  const { center, angle, half } = best;
  const c = Math.cos(angle), s = Math.sin(angle);
  const scale = Math.min(half[0], half[1]);
  const errors = side.map((p) => {
    const dx = p[0] - center[0], dz = p[2] - center[1];
    const u = Math.abs(c * dx + s * dz), v = Math.abs(-s * dx + c * dz);
    // distance to the nearest face of the box's outline
    const outside = Math.hypot(Math.max(0, u - half[0]), Math.max(0, v - half[1]));
    return (outside > 0 ? outside : Math.min(half[0] - u, half[1] - v)) / scale;
  });
  return {
    kind: "box", center, angle, half, bottom, top,
    inliers: errors.filter((e) => e < tolerance).length / errors.length,
    residual: median(errors),
  };
}

/**
 * Above this, a surface is shiny (see scan-shine.ts). Measured: a synthetic
 * matte orbit 0.12, a real aluminium can 0.35.
 */
export const SHINY = 0.22;

/** The shape to put in place of the scanned surface, or null to keep the scan. */
export function chooseShape(sample: SurfaceSample, shine: number): ShapeFit | null {
  const shiny = shine >= SHINY;
  const revolve = fitRevolve(sample, { envelope: shiny ? 0.75 : 0.5 });
  const box = fitBox(sample);
  const round = revolve && revolve.protrusion < 0.03 ? revolve : null;
  if (shiny) {
    // dents cannot be told from a narrower part of a box, so a shiny scan only
    // becomes round, and only if a box does not explain it better (a can:
    // 0.10 against 0.21; a box with a ball on it: 0.087 against 0.057)
    return round && round.residual < 0.15 && (!box || round.residual < box.residual) ? round : null;
  }
  const fits = [round, box].filter((fit): fit is ShapeFit => !!fit && fit.inliers >= 0.85);
  return fits.sort((a, b) => b.inliers - a.inliers)[0] ?? null;
}

/** Quick test before the photos are read: could either shape fit at all? */
export function mightBeShape(sample: SurfaceSample) {
  const revolve = fitRevolve(sample, { envelope: 0.75 });
  const box = fitBox(sample);
  return (!!revolve && revolve.residual < 0.15 && revolve.protrusion < 0.03) || (!!box && box.inliers >= 0.85);
}

/** A closed, outward-facing mesh of the shape, with the caps kept apart for crisp edges. */
export function shapeMesh(fit: ShapeFit, segments = 96) {
  const positions: number[] = [];
  const indices: number[] = [];
  const vertex = (x: number, y: number, z: number) => (positions.push(x, y, z), positions.length / 3 - 1);
  // a grid of nu x nv quads through point(i, j), facing (∂point/∂i) × (∂point/∂j)
  const face = (point: (i: number, j: number) => Vec3, nu: number, nv: number) => {
    const base = positions.length / 3;
    for (let j = 0; j <= nv; j++) for (let i = 0; i <= nu; i++) vertex(...point(i, j));
    for (let j = 0; j < nv; j++) for (let i = 0; i < nu; i++) {
      const a = base + j * (nu + 1) + i, b = a + 1, c = a + nu + 1, d = c + 1;
      indices.push(a, b, c, b, d, c);
    }
  };
  const { bottom, top } = fit;
  if (fit.kind === "revolve") {
    const { profile, center: [cx, cz] } = fit;
    const rings = profile.length;
    // radius at height y, between the band middles
    const radiusAt = (y: number) => {
      const position = Math.min(rings - 1, Math.max(0, (y - bottom) / (top - bottom) * rings - 0.5));
      const lo = Math.floor(position), hi = Math.min(rings - 1, lo + 1);
      return profile[lo] + (profile[hi] - profile[lo]) * (position - lo);
    };
    const around = (i: number, y: number, r: number): Vec3 => {
      const a = (i % segments) / segments * 2 * Math.PI;
      return [cx + r * Math.cos(a), y, cz - r * Math.sin(a)];
    };
    // the side shares its seam so its normals stay smooth all the way round
    const base = positions.length / 3;
    for (let j = 0; j <= rings; j++) {
      const y = bottom + (top - bottom) * j / rings;
      for (let i = 0; i < segments; i++) vertex(...around(i, y, radiusAt(y)));
    }
    for (let j = 0; j < rings; j++) for (let i = 0; i < segments; i++) {
      const a = base + j * segments + i, b = base + j * segments + (i + 1) % segments;
      const c = a + segments, d = b + segments;
      indices.push(a, b, c, b, d, c);
    }
    for (const [y, up] of [[bottom, false], [top, true]] as const) {
      const middle = vertex(cx, y, cz);
      const rim = positions.length / 3;
      for (let i = 0; i < segments; i++) vertex(...around(i, y, radiusAt(y)));
      for (let i = 0; i < segments; i++) {
        const a = rim + i, b = rim + (i + 1) % segments;
        indices.push(...(up ? [middle, a, b] : [middle, b, a]));
      }
    }
  } else {
    const { center: [cx, cz], angle, half: [hu, hv] } = fit;
    const c = Math.cos(angle), s = Math.sin(angle);
    // box axes u, v back to x, z
    const at = (u: number, y: number, v: number): Vec3 => [cx + c * u - s * v, y, cz + s * u + c * v];
    const height = top - bottom;
    const step = Math.max(hu, hv, height / 2) / 12;
    const nu = Math.max(2, Math.round(2 * hu / step)), nv = Math.max(2, Math.round(2 * hv / step)), ny = Math.max(2, Math.round(height / step));
    const lerp = (a: number, b: number, k: number, n: number) => a + (b - a) * k / n;
    face((i, j) => at(lerp(-hu, hu, i, nu), lerp(bottom, top, j, ny), hv), nu, ny);
    face((i, j) => at(hu, lerp(bottom, top, j, ny), lerp(hv, -hv, i, nv)), nv, ny);
    face((i, j) => at(lerp(hu, -hu, i, nu), lerp(bottom, top, j, ny), -hv), nu, ny);
    face((i, j) => at(-hu, lerp(bottom, top, j, ny), lerp(-hv, hv, i, nv)), nv, ny);
    face((i, j) => at(lerp(-hu, hu, i, nu), top, lerp(hv, -hv, j, nv)), nu, nv);
    face((i, j) => at(lerp(-hu, hu, i, nu), bottom, lerp(-hv, hv, j, nv)), nu, nv);
  }
  return { positions: Float32Array.from(positions), indices: Uint32Array.from(indices) };
}

/** The scanned mesh with its surface replaced by the shape: same frame, colors from the splat. */
export function snapToShape(mesh: ScanMesh, fit: ShapeFit, targetTriangles?: number): ScanMesh {
  let { positions, indices } = shapeMesh(fit);
  if (targetTriangles && indices.length / 3 > targetTriangles) ({ positions, indices } = decimate(positions, indices, targetTriangles));
  const count = positions.length / 3;
  const colors = new Float32Array(count * 3);
  const color: Vec3 = [0, 0, 0];
  const min: Vec3 = [Infinity, Infinity, Infinity], max: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (let v = 0; v < count; v++) {
    const p: Vec3 = [positions[v * 3], positions[v * 3 + 1], positions[v * 3 + 2]];
    colors.set(mesh.sampleColor(p, color), v * 3);
    for (let k = 0; k < 3; k++) { min[k] = Math.min(min[k], p[k]); max[k] = Math.max(max[k], p[k]); }
  }
  return {
    ...mesh,
    positions,
    normals: computeNormals(positions, indices),
    colors,
    indices,
    stats: { ...mesh.stats, triangles: indices.length / 3, sizeMeters: [max[0] - min[0], max[1] - min[1], max[2] - min[2]], shape: fit.kind },
  };
}

/** Explicit user-supplied dimensions, not a fitted or measured surface. */
export function dimensionedCylinder(height: number, diameter: number): RevolveFit {
  if (!Number.isFinite(height) || !Number.isFinite(diameter) || height <= 0 || diameter <= 0) {
    throw new RangeError("Enter a positive measured height and diameter.");
  }
  return { kind: "revolve", center: [0, 0], bottom: 0, top: height,
    profile: new Float32Array(2).fill(diameter / 2), inliers: 0, residual: 0, protrusion: 0 };
}
