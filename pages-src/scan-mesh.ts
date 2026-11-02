// Turns a trained Gaussian splat into a watertight, colored triangle mesh that
// every AR viewer understands. Everything here is plain typed-array math so it
// runs on the phone that made the scan and can be unit tested in Node.

import { edgeTable, triTable } from "three/addons/objects/MarchingCubes.js";

export type Vec3 = [number, number, number];

/** splat.js trainer layout, 16 floats per splat:
 *  0-2 position, 3-5 log scale, 6-9 quaternion (w, x, y, z), 10-12 color, 13 logit opacity. */
export interface SplatCloud {
  data: Float32Array;
  n: number;
  dc: "sigmoid" | "sh";
}

export interface ScanCamera {
  position: Vec3;
  forward: Vec3;
  up: Vec3;
}

export interface MeshOptions {
  /** voxels along the longest side of the subject */
  resolution?: number;
  /** subject radius as a fraction of the cameras' median distance from the orbit axis */
  crop?: number;
  /** real-world height of the subject in meters */
  heightMeters?: number;
  /** density threshold, relative to typical surface density */
  density?: number;
  removeGround?: boolean;
  smoothing?: number;
}

export interface ScanMesh {
  positions: Float32Array;
  normals: Float32Array;
  colors: Float32Array;
  indices: Uint32Array;
  stats: {
    splatsUsed: number;
    groundRemoved: boolean;
    voxels: Vec3;
    triangles: number;
    sizeMeters: Vec3;
  };
}

const STRIDE = 16;
const SH_C0 = 0.28209479177387814;
const sigmoid = (value: number) => 1 / (1 + Math.exp(-value));

const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const scale = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];
const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const length = (a: Vec3) => Math.hypot(a[0], a[1], a[2]);
const normalize = (a: Vec3): Vec3 => {
  const l = length(a);
  return l > 1e-12 ? scale(a, 1 / l) : [0, 0, 0];
};

function median(values: number[]) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[sorted.length >> 1] ?? 0;
}

function quantile(values: Float32Array | number[], q: number) {
  const sorted = Float32Array.from(values).sort();
  return sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))] ?? 0;
}

/** The point closest to every camera's optical axis: what an orbit looks at. */
export function orbitCenter(cameras: ScanCamera[]): Vec3 | null {
  if (cameras.length < 3) return null;
  const a = new Float64Array(9);
  const b = new Float64Array(3);
  for (const camera of cameras) {
    const d = normalize(camera.forward);
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        const m = (r === c ? 1 : 0) - d[r] * d[c];
        a[r * 3 + c] += m;
        b[r] += m * camera.position[c];
      }
    }
  }
  const det = (m: ArrayLike<number>) =>
    m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
  const d = det(a);
  // parallel axes (a forward walk, not an orbit) leave the system singular
  if (Math.abs(d) < 1e-3 * cameras.length ** 3) return null;
  const solved: number[] = [];
  for (let k = 0; k < 3; k++) {
    const m = Array.from(a);
    for (let r = 0; r < 3; r++) m[r * 3 + k] = b[r];
    solved.push(det(m) / d);
  }
  const center = solved as Vec3;
  const inFront = cameras.filter((camera) => dot(sub(center, camera.position), camera.forward) > 0).length;
  return inFront >= 0.7 * cameras.length ? center : null;
}

/** Rotation taking unit vector `from` onto unit vector `to` (row-major 3x3). */
function rotationBetween(from: Vec3, to: Vec3) {
  const v = cross(from, to);
  const c = dot(from, to);
  if (c < -0.999999) return [1, 0, 0, 0, -1, 0, 0, 0, -1];
  const k = 1 / (1 + c);
  return [
    v[0] * v[0] * k + c, v[0] * v[1] * k - v[2], v[0] * v[2] * k + v[1],
    v[1] * v[0] * k + v[2], v[1] * v[1] * k + c, v[1] * v[2] * k - v[0],
    v[2] * v[0] * k - v[1], v[2] * v[1] * k + v[0], v[2] * v[2] * k + c,
  ];
}

const apply = (m: number[], p: Vec3): Vec3 => [
  m[0] * p[0] + m[1] * p[1] + m[2] * p[2],
  m[3] * p[0] + m[4] * p[1] + m[5] * p[2],
  m[6] * p[0] + m[7] * p[1] + m[8] * p[2],
];

/** Dominant near-horizontal plane under the subject (the table or floor). */
function findGround(points: Vec3[], tolerance: number) {
  if (points.length < 50) return null;
  const sample = points.length > 4000 ? Array.from({ length: 4000 }, (_, i) => points[Math.floor(i * points.length / 4000)]) : points;
  let seed = 12345;
  const random = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
  let best: { normal: Vec3; offset: number; inliers: number } | null = null;
  for (let attempt = 0; attempt < 400; attempt++) {
    const p0 = sample[Math.floor(random() * sample.length)];
    const p1 = sample[Math.floor(random() * sample.length)];
    const p2 = sample[Math.floor(random() * sample.length)];
    let normal = normalize(cross(sub(p1, p0), sub(p2, p0)));
    if (normal[1] < 0) normal = scale(normal, -1);
    if (normal[1] < Math.cos(25 * Math.PI / 180)) continue;
    const offset = dot(normal, p0);
    if (offset >= 0) continue; // the ground has to sit below the orbit center
    let inliers = 0;
    for (const p of sample) if (Math.abs(dot(normal, p) - offset) < tolerance) inliers++;
    if (!best || inliers > best.inliers) best = { normal, offset, inliers };
  }
  return best && best.inliers >= 0.08 * sample.length ? best : null;
}

interface PreparedSplat {
  position: Vec3;
  axes: [Vec3, Vec3, Vec3];
  sigmas: Vec3;
  alpha: number;
  color: Vec3;
}

/** Crop the splat to the scanned subject in an upright, ground-aligned frame. */
export function prepareSubject(cloud: SplatCloud, cameras: ScanCamera[], options: MeshOptions = {}) {
  const crop = options.crop ?? 0.45;
  const { data, n } = cloud;

  let up = normalize(cameras.reduce<Vec3>((sum, camera) => [sum[0] + camera.up[0], sum[1] + camera.up[1], sum[2] + camera.up[2]], [0, 0, 0]));
  if (length(up) < 0.5) up = normalize(cameras[0]?.up ?? [0, -1, 0]);

  let center = orbitCenter(cameras);
  if (!center) {
    // no orbit: the opacity-weighted median of the splats is the best guess
    const xs: number[] = [], ys: number[] = [], zs: number[] = [];
    for (let i = 0; i < n; i++) {
      if (sigmoid(data[i * STRIDE + 13]) < 0.5) continue;
      xs.push(data[i * STRIDE]); ys.push(data[i * STRIDE + 1]); zs.push(data[i * STRIDE + 2]);
    }
    center = [median(xs), median(ys), median(zs)];
  }
  const reach = median(cameras.map((camera) => length(sub(camera.position, center!)))) || 1;

  // basis: +Y up, +Z toward the first camera so the first photo is the front view
  let toward = sub(cameras[0]?.position ?? [0, 0, -1], center);
  toward = normalize(sub(toward, scale(up, dot(toward, up))));
  if (length(toward) < 0.5) toward = normalize(cross(up, [1, 0, 0]));
  const xAxis = normalize(cross(up, toward));
  const basis = [...xAxis, ...up, ...toward];

  // an orbit keeps the subject inside the ring of cameras: crop a cylinder
  // around the vertical axis through the orbit center, sized by that ring
  const ring = median(cameras.map((camera) => {
    const local = apply(basis, sub(camera.position, center!));
    return Math.hypot(local[0], local[2]);
  })) || reach;
  const radius = crop * ring;
  const sh = cloud.dc === "sh";
  const splats: PreparedSplat[] = [];
  for (let i = 0; i < n; i++) {
    const o = i * STRIDE;
    const alpha = sigmoid(data[o + 13]);
    if (alpha < 0.15) continue;
    const local = apply(basis, sub([data[o], data[o + 1], data[o + 2]], center));
    if (Math.hypot(local[0], local[2]) > radius || Math.abs(local[1]) > reach) continue;
    let qw = data[o + 6], qx = data[o + 7], qy = data[o + 8], qz = data[o + 9];
    const ql = Math.hypot(qw, qx, qy, qz) || 1;
    qw /= ql; qx /= ql; qy /= ql; qz /= ql;
    // columns of the quaternion's rotation matrix are the splat's principal axes
    const axes: [Vec3, Vec3, Vec3] = [
      apply(basis, [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy + qw * qz), 2 * (qx * qz - qw * qy)]),
      apply(basis, [2 * (qx * qy - qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz + qw * qx)]),
      apply(basis, [2 * (qx * qz + qw * qy), 2 * (qy * qz - qw * qx), 1 - 2 * (qx * qx + qy * qy)]),
    ];
    const color: Vec3 = sh
      ? [0.5 + SH_C0 * data[o + 10], 0.5 + SH_C0 * data[o + 11], 0.5 + SH_C0 * data[o + 12]]
      : [sigmoid(data[o + 10]), sigmoid(data[o + 11]), sigmoid(data[o + 12])];
    splats.push({
      position: local,
      axes,
      sigmas: [Math.exp(data[o + 3]), Math.exp(data[o + 4]), Math.exp(data[o + 5])],
      alpha,
      color: color.map((c) => Math.min(1, Math.max(0, c))) as Vec3,
    });
  }

  let floor = -Infinity;
  let groundRemoved = false;
  if (options.removeGround !== false) {
    const solid = splats.filter((splat) => splat.alpha > 0.5).map((splat) => splat.position);
    // a trained splat surface is a few percent of the subject radius thick
    const ground = findGround(solid, 0.04 * radius);
    if (ground) {
      // level the subject on the ground plane, then cut the plane away
      const level = rotationBetween(ground.normal, [0, 1, 0]);
      for (const splat of splats) {
        splat.position = apply(level, splat.position);
        splat.axes = splat.axes.map((axis) => apply(level, axis)) as [Vec3, Vec3, Vec3];
      }
      floor = ground.offset + 0.06 * radius;
      groundRemoved = true;
    }
  }
  const subject = splats.filter((splat) => splat.position[1] > floor);
  return { splats: subject, radius, floor, groundRemoved };
}

interface Grid {
  dims: Vec3;
  origin: Vec3;
  cell: number;
}

const index = (grid: Grid, x: number, y: number, z: number) => (z * grid.dims[1] + y) * grid.dims[0] + x;

function blur(field: Float32Array, dims: Vec3) {
  // separable [1 2 1] / 4 along each axis
  const [nx, ny, nz] = dims;
  let src = field;
  let dst = new Float32Array(field.length);
  const strides = [1, nx, nx * ny];
  const sizes = [nx, ny, nz];
  for (let axis = 0; axis < 3; axis++) {
    const s = strides[axis];
    for (let z = 0; z < nz; z++) {
      for (let y = 0; y < ny; y++) {
        for (let x = 0; x < nx; x++) {
          const i = (z * ny + y) * nx + x;
          const coordinate = axis === 0 ? x : axis === 1 ? y : z;
          const before = coordinate > 0 ? src[i - s] : 0;
          const after = coordinate < sizes[axis] - 1 ? src[i + s] : 0;
          dst[i] = 0.25 * before + 0.5 * src[i] + 0.25 * after;
        }
      }
    }
    [src, dst] = [dst, src];
  }
  return src;
}

const NEIGHBORS: Vec3[] = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];

/** Breadth-first layers out from `seeds` through voxels `passable` allows,
 *  up to `limit` steps; returns the step count per voxel (255 = unreached). */
function bfsLayers(grid: Grid, seeds: Int32Array | number[], passable: (i: number) => boolean, limit = 254) {
  const [nx, ny, nz] = grid.dims;
  const depth = new Uint8Array(nx * ny * nz).fill(255);
  let frontier = Array.from(seeds);
  for (const i of frontier) depth[i] = 0;
  for (let step = 1; step <= limit && frontier.length; step++) {
    const next: number[] = [];
    for (const i of frontier) {
      const x = i % nx, y = Math.floor(i / nx) % ny, z = Math.floor(i / (nx * ny));
      for (const [dx, dy, dz] of NEIGHBORS) {
        const X = x + dx, Y = y + dy, Z = z + dz;
        if (X < 0 || Y < 0 || Z < 0 || X >= nx || Y >= ny || Z >= nz) continue;
        const j = index(grid, X, Y, Z);
        if (depth[j] !== 255 || !passable(j)) continue;
        depth[j] = step;
        next.push(j);
      }
    }
    frontier = next;
  }
  return depth;
}

/** Fill the enclosed interior of a voxel shell, drop floaters, return a solid mask.
 *  `floorLayer` (>= 0) seals the shell from below: nobody photographs the
 *  underside of an object resting on a table, so scans are open at the bottom. */
function solidify(occupied: Uint8Array, grid: Grid, floorLayer: number, closing = 2) {
  const [nx, ny, nz] = grid.dims;
  const total = nx * ny * nz;
  const occupiedList: number[] = [];
  for (let i = 0; i < total; i++) if (occupied[i]) occupiedList.push(i);

  // close holes up to `closing` voxels wide before the flood so it cannot leak inside
  const grown = bfsLayers(grid, occupiedList, () => true, closing);
  const isFloor = (i: number) => floorLayer >= 0 && Math.floor(i / nx) % ny === floorLayer;
  const closed = (i: number) => grown[i] !== 255 || isFloor(i);

  const border: number[] = [];
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    if (x && y && z && x < nx - 1 && y < ny - 1 && z < nz - 1) continue;
    const i = index(grid, x, y, z);
    if (!closed(i) && !(floorLayer >= 0 && y < floorLayer)) border.push(i);
  }
  const outsideDepth = bfsLayers(grid, border, (i) => !closed(i) && !(floorLayer >= 0 && Math.floor(i / nx) % ny < floorLayer));
  const outside = (i: number) => outsideDepth[i] !== 255;

  // undo the closing on the outer surface: peel the added layers back off
  const shellSeeds: number[] = [];
  for (let i = 0; i < total; i++) if (outside(i)) shellSeeds.push(i);
  const fromOutside = bfsLayers(grid, shellSeeds, (i) => !outside(i), closing);
  const solid = new Uint8Array(total);
  for (let i = 0; i < total; i++) {
    if (outside(i) || (floorLayer >= 0 && Math.floor(i / nx) % ny < floorLayer)) continue;
    solid[i] = occupied[i] || fromOutside[i] === 255 ? 1 : 0;
  }
  // the seal only stays where it caps the subject
  if (floorLayer >= 0 && floorLayer + 1 < ny) {
    for (let z = 0; z < nz; z++) for (let x = 0; x < nx; x++) {
      const i = index(grid, x, floorLayer, z);
      if (!solid[index(grid, x, floorLayer + 1, z)]) solid[i] = 0;
    }
  }

  // opening: erode one voxel so thin sheets (leftover table, wall slivers)
  // detach from the body, pick the body, then grow it back inside the solid
  const eroded = new Uint8Array(total);
  for (let z = 1; z < nz - 1; z++) for (let y = 1; y < ny - 1; y++) for (let x = 1; x < nx - 1; x++) {
    const i = index(grid, x, y, z);
    if (!solid[i]) continue;
    let inner = 1;
    for (const [dx, dy, dz] of NEIGHBORS) if (!solid[index(grid, x + dx, y + dy, z + dz)]) { inner = 0; break; }
    eroded[i] = inner;
  }
  const body = largestParts(eroded, grid);
  const seeds: number[] = [];
  for (let i = 0; i < total; i++) if (body[i]) seeds.push(i);
  const regrown = bfsLayers(grid, seeds, (i) => solid[i] === 1, 1);
  for (let i = 0; i < total; i++) solid[i] = regrown[i] !== 255 ? 1 : 0;
  return solid;
}

/** Keep the largest connected part and any part at least 15% of its size. */
function largestParts(solid: Uint8Array, grid: Grid) {
  const [nx, ny, nz] = grid.dims;
  const total = nx * ny * nz;
  const label = new Int32Array(total).fill(-1);
  const queue = new Int32Array(total);
  const sizes: number[] = [];
  for (let start = 0; start < total; start++) {
    if (!solid[start] || label[start] >= 0) continue;
    const id = sizes.length;
    let head = 0, tail = 0;
    queue[tail++] = start;
    label[start] = id;
    while (head < tail) {
      const i = queue[head++];
      const x = i % nx, y = Math.floor(i / nx) % ny, z = Math.floor(i / (nx * ny));
      for (const [dx, dy, dz] of NEIGHBORS) {
        const X = x + dx, Y = y + dy, Z = z + dz;
        if (X < 0 || Y < 0 || Z < 0 || X >= nx || Y >= ny || Z >= nz) continue;
        const j = index(grid, X, Y, Z);
        if (!solid[j] || label[j] >= 0) continue;
        label[j] = id;
        queue[tail++] = j;
      }
    }
    sizes.push(tail);
  }
  const largest = Math.max(0, ...sizes);
  for (let i = 0; i < total; i++) if (solid[i] && sizes[label[i]] < 0.15 * largest) solid[i] = 0;
  return solid;
}

/** Marching cubes over a scalar field; triangles face away from high values. */
function marchingCubes(field: Float32Array, grid: Grid, iso: number) {
  const [nx, ny, nz] = grid.dims;
  const edgeVertex = new Int32Array(nx * ny * nz * 3).fill(-1);
  const positions: number[] = [];
  const indices: number[] = [];
  // edge -> [corner offset, axis] in the corner order v0..v7 of the tables
  const corners: Vec3[] = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]];
  const edges: [number, number][] = [[0, 0], [1, 1], [3, 0], [0, 1], [4, 0], [5, 1], [7, 0], [4, 1], [0, 2], [1, 2], [2, 2], [3, 2]];
  const cornerValues = new Float32Array(8);
  const cubeEdges = new Int32Array(12);

  for (let z = 0; z < nz - 1; z++) for (let y = 0; y < ny - 1; y++) for (let x = 0; x < nx - 1; x++) {
    let cube = 0;
    for (let c = 0; c < 8; c++) {
      const value = field[index(grid, x + corners[c][0], y + corners[c][1], z + corners[c][2])];
      cornerValues[c] = value;
      if (value < iso) cube |= 1 << c;
    }
    const mask = edgeTable[cube];
    if (!mask) continue;
    for (let e = 0; e < 12; e++) {
      if (!(mask & (1 << e))) continue;
      const [corner, axis] = edges[e];
      const X = x + corners[corner][0], Y = y + corners[corner][1], Z = z + corners[corner][2];
      const key = index(grid, X, Y, Z) * 3 + axis;
      if (edgeVertex[key] < 0) {
        const a = field[index(grid, X, Y, Z)];
        const b = field[index(grid, X + (axis === 0 ? 1 : 0), Y + (axis === 1 ? 1 : 0), Z + (axis === 2 ? 1 : 0))];
        const t = Math.abs(b - a) > 1e-9 ? (iso - a) / (b - a) : 0.5;
        const p: Vec3 = [X, Y, Z];
        p[axis] += t;
        edgeVertex[key] = positions.length / 3;
        positions.push(
          grid.origin[0] + p[0] * grid.cell,
          grid.origin[1] + p[1] * grid.cell,
          grid.origin[2] + p[2] * grid.cell,
        );
      }
      cubeEdges[e] = edgeVertex[key];
    }
    const row = cube << 4;
    for (let k = 0; triTable[row + k] !== -1; k += 3) {
      indices.push(cubeEdges[triTable[row + k]], cubeEdges[triTable[row + k + 1]], cubeEdges[triTable[row + k + 2]]);
    }
  }
  return { positions: new Float32Array(positions), indices: new Uint32Array(indices) };
}

/** Taubin smoothing: removes voxel stair-steps without shrinking the model. */
function smooth(positions: Float32Array, indices: Uint32Array, iterations: number) {
  const count = positions.length / 3;
  const neighborSets: Set<number>[] = Array.from({ length: count }, () => new Set());
  for (let t = 0; t < indices.length; t += 3) {
    const a = indices[t], b = indices[t + 1], c = indices[t + 2];
    neighborSets[a].add(b).add(c);
    neighborSets[b].add(a).add(c);
    neighborSets[c].add(a).add(b);
  }
  const neighbors = neighborSets.map((set) => Int32Array.from(set));
  const next = new Float32Array(positions.length);
  for (let iteration = 0; iteration < iterations * 2; iteration++) {
    const factor = iteration % 2 === 0 ? 0.5 : -0.53;
    for (let v = 0; v < count; v++) {
      const list = neighbors[v];
      for (let k = 0; k < 3; k++) {
        if (!list.length) { next[v * 3 + k] = positions[v * 3 + k]; continue; }
        let mean = 0;
        for (const j of list) mean += positions[j * 3 + k];
        mean /= list.length;
        next[v * 3 + k] = positions[v * 3 + k] + factor * (mean - positions[v * 3 + k]);
      }
    }
    positions.set(next);
  }
}

function computeNormals(positions: Float32Array, indices: Uint32Array) {
  const normals = new Float32Array(positions.length);
  for (let t = 0; t < indices.length; t += 3) {
    const a = indices[t] * 3, b = indices[t + 1] * 3, c = indices[t + 2] * 3;
    const ab: Vec3 = [positions[b] - positions[a], positions[b + 1] - positions[a + 1], positions[b + 2] - positions[a + 2]];
    const ac: Vec3 = [positions[c] - positions[a], positions[c + 1] - positions[a + 1], positions[c + 2] - positions[a + 2]];
    const face = cross(ab, ac);
    for (const v of [a, b, c]) { normals[v] += face[0]; normals[v + 1] += face[1]; normals[v + 2] += face[2]; }
  }
  for (let v = 0; v < normals.length; v += 3) {
    const l = Math.hypot(normals[v], normals[v + 1], normals[v + 2]) || 1;
    normals[v] /= l; normals[v + 1] /= l; normals[v + 2] /= l;
  }
  return normals;
}

export function buildScanMesh(cloud: SplatCloud, cameras: ScanCamera[], options: MeshOptions = {}): ScanMesh {
  const resolution = options.resolution ?? 96;
  const { splats, groundRemoved, floor } = prepareSubject(cloud, cameras, options);
  if (splats.length < 50) throw new Error("Too little of the scan landed on the subject. Widen the crop or rescan closer to the object.");

  // grid around the robust extent of the subject (ignores the last stray splats)
  const lo: Vec3 = [0, 0, 0], hi: Vec3 = [0, 0, 0];
  for (let k = 0; k < 3; k++) {
    const values = splats.map((splat) => splat.position[k]);
    lo[k] = quantile(values, 0.002);
    hi[k] = quantile(values, 0.998);
  }
  if (groundRemoved) lo[1] = Math.max(lo[1], floor);
  const extent = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) || 1;
  const cell = extent / (resolution - 6);
  const grid: Grid = {
    cell,
    origin: [lo[0] - 3 * cell, lo[1] - 3 * cell, lo[2] - 3 * cell],
    dims: [0, 1, 2].map((k) => Math.ceil((hi[k] - lo[k]) / cell) + 7) as Vec3,
  };
  const [nx, ny, nz] = grid.dims;
  const total = nx * ny * nz;
  const density = new Float32Array(total);
  const colorSum = new Float32Array(total * 3);
  const weightSum = new Float32Array(total);

  const deposit = (i: number, weight: number, color: Vec3) => {
    density[i] += weight;
    weightSum[i] += weight;
    colorSum[i * 3] += weight * color[0];
    colorSum[i * 3 + 1] += weight * color[1];
    colorSum[i * 3 + 2] += weight * color[2];
  };

  let used = 0;
  for (const splat of splats) {
    const g: Vec3 = [
      (splat.position[0] - grid.origin[0]) / cell,
      (splat.position[1] - grid.origin[1]) / cell,
      (splat.position[2] - grid.origin[2]) / cell,
    ];
    if (g.some((value, k) => value < 0 || value >= grid.dims[k] - 1)) continue;
    used++;
    const largest = Math.max(...splat.sigmas);
    if (largest < 0.75 * cell) {
      // sub-voxel splat: trilinear deposit
      const x0 = Math.floor(g[0]), y0 = Math.floor(g[1]), z0 = Math.floor(g[2]);
      const fx = g[0] - x0, fy = g[1] - y0, fz = g[2] - z0;
      for (let c = 0; c < 8; c++) {
        const dx = c & 1, dy = (c >> 1) & 1, dz = (c >> 2) & 1;
        const w = (dx ? fx : 1 - fx) * (dy ? fy : 1 - fy) * (dz ? fz : 1 - fz);
        deposit(index(grid, x0 + dx, y0 + dy, z0 + dz), splat.alpha * w, splat.color);
      }
      continue;
    }
    // larger splat: evaluate its anisotropic Gaussian over the voxels it covers
    const sigmas = splat.sigmas.map((s) => Math.max(s, 0.35 * cell)) as Vec3;
    const reach = Math.min(4, Math.ceil((2.5 * largest) / cell));
    const cx = Math.round(g[0]), cy = Math.round(g[1]), cz = Math.round(g[2]);
    for (let z = Math.max(0, cz - reach); z <= Math.min(nz - 1, cz + reach); z++) {
      for (let y = Math.max(0, cy - reach); y <= Math.min(ny - 1, cy + reach); y++) {
        for (let x = Math.max(0, cx - reach); x <= Math.min(nx - 1, cx + reach); x++) {
          const d: Vec3 = [(x - g[0]) * cell, (y - g[1]) * cell, (z - g[2]) * cell];
          let m = 0;
          for (let k = 0; k < 3; k++) {
            const projected = dot(splat.axes[k], d) / sigmas[k];
            m += projected * projected;
          }
          if (m > 9) continue;
          deposit(index(grid, x, y, z), splat.alpha * Math.exp(-0.5 * m), splat.color);
        }
      }
    }
  }

  // threshold relative to how dense the subject's surface actually is
  const blurred = blur(density, grid.dims);
  const filled: number[] = [];
  for (let i = 0; i < total; i++) if (blurred[i] > 0.02) filled.push(blurred[i]);
  const iso = Math.max(0.05, (options.density ?? 0.35) * quantile(filled, 0.8));
  const occupied = new Uint8Array(total);
  const floorVoxel = groundRemoved ? Math.max(1, Math.ceil((floor - grid.origin[1]) / cell)) : -1;
  for (let i = 0; i < total; i++) {
    const y = Math.floor(i / nx) % ny;
    if (blurred[i] > iso && y >= floorVoxel) occupied[i] = 1;
  }
  const solid = solidify(occupied, grid, floorVoxel);

  const field = new Float32Array(total);
  for (let i = 0; i < total; i++) field[i] = solid[i];
  const smoothField = blur(field, grid.dims);
  const { positions, indices } = marchingCubes(smoothField, grid, 0.5);
  if (!indices.length) throw new Error("No surface was found. Try a lower density or a wider crop.");
  smooth(positions, indices, options.smoothing ?? 3);

  // spread surface colors outward a few voxels so every vertex finds one
  const color = new Float32Array(total * 3);
  const known = new Uint8Array(total);
  for (let i = 0; i < total; i++) {
    if (weightSum[i] < 1e-4) continue;
    known[i] = 1;
    color[i * 3] = colorSum[i * 3] / weightSum[i];
    color[i * 3 + 1] = colorSum[i * 3 + 1] / weightSum[i];
    color[i * 3 + 2] = colorSum[i * 3 + 2] / weightSum[i];
  }
  for (let pass = 0; pass < 4; pass++) {
    const additions: number[] = [];
    for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
      const i = index(grid, x, y, z);
      if (known[i]) continue;
      let r = 0, g = 0, b = 0, c = 0;
      for (const [dx, dy, dz] of NEIGHBORS) {
        const X = x + dx, Y = y + dy, Z = z + dz;
        if (X < 0 || Y < 0 || Z < 0 || X >= nx || Y >= ny || Z >= nz) continue;
        const j = index(grid, X, Y, Z);
        if (!known[j]) continue;
        r += color[j * 3]; g += color[j * 3 + 1]; b += color[j * 3 + 2]; c++;
      }
      if (c) additions.push(i, r / c, g / c, b / c);
    }
    for (let k = 0; k < additions.length; k += 4) {
      const i = additions[k];
      known[i] = 1;
      color[i * 3] = additions[k + 1]; color[i * 3 + 1] = additions[k + 2]; color[i * 3 + 2] = additions[k + 3];
    }
  }

  const vertexCount = positions.length / 3;
  const colors = new Float32Array(vertexCount * 3).fill(0.6);
  for (let v = 0; v < vertexCount; v++) {
    const g: Vec3 = [
      (positions[v * 3] - grid.origin[0]) / cell,
      (positions[v * 3 + 1] - grid.origin[1]) / cell,
      (positions[v * 3 + 2] - grid.origin[2]) / cell,
    ];
    const x0 = Math.min(nx - 2, Math.max(0, Math.floor(g[0])));
    const y0 = Math.min(ny - 2, Math.max(0, Math.floor(g[1])));
    const z0 = Math.min(nz - 2, Math.max(0, Math.floor(g[2])));
    let r = 0, gr = 0, b = 0, w = 0;
    for (let c = 0; c < 8; c++) {
      const dx = c & 1, dy = (c >> 1) & 1, dz = (c >> 2) & 1;
      const i = index(grid, x0 + dx, y0 + dy, z0 + dz);
      if (!known[i]) continue;
      const weight = Math.max(1e-4, (1 - Math.abs(g[0] - x0 - dx)) * (1 - Math.abs(g[1] - y0 - dy)) * (1 - Math.abs(g[2] - z0 - dz)))
        * (0.25 + Math.min(1, weightSum[i])); // prefer voxels the splats actually colored
      r += weight * color[i * 3]; gr += weight * color[i * 3 + 1]; b += weight * color[i * 3 + 2]; w += weight;
    }
    if (w > 0) { colors[v * 3] = r / w; colors[v * 3 + 1] = gr / w; colors[v * 3 + 2] = b / w; }
  }

  // meters, centered on the floor footprint, resting on y = 0
  const min: Vec3 = [Infinity, Infinity, Infinity], max: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (let v = 0; v < vertexCount; v++) {
    for (let k = 0; k < 3; k++) {
      min[k] = Math.min(min[k], positions[v * 3 + k]);
      max[k] = Math.max(max[k], positions[v * 3 + k]);
    }
  }
  const height = max[1] - min[1] || 1;
  const metersPerUnit = (options.heightMeters ?? 0.3) / height;
  const middle: Vec3 = [(min[0] + max[0]) / 2, min[1], (min[2] + max[2]) / 2];
  for (let v = 0; v < vertexCount; v++) {
    for (let k = 0; k < 3; k++) positions[v * 3 + k] = (positions[v * 3 + k] - middle[k]) * metersPerUnit;
  }

  return {
    positions,
    normals: computeNormals(positions, indices),
    colors,
    indices,
    stats: {
      splatsUsed: used,
      groundRemoved,
      voxels: grid.dims,
      triangles: indices.length / 3,
      sizeMeters: [(max[0] - min[0]) * metersPerUnit, height * metersPerUnit, (max[2] - min[2]) * metersPerUnit],
    },
  };
}

/** Bake per-vertex colors into a texture atlas: each triangle owns a 2x2 texel
 *  cell whose texel centers carry its corner colors, so bilinear sampling
 *  reproduces the vertex-color gradient. Apple Quick Look ignores vertex colors
 *  on textured materials; a texture survives every converter. */
export function bakeColorAtlas(mesh: ScanMesh) {
  const triangles = mesh.indices.length / 3;
  const cellsPerRow = Math.ceil(Math.sqrt(triangles));
  let size = 64;
  while (size < cellsPerRow * 2) size *= 2;
  const pixels = new Uint8ClampedArray(size * size * 4);
  const positions = new Float32Array(triangles * 9);
  const normals = new Float32Array(triangles * 9);
  const uvs = new Float32Array(triangles * 6);
  const toByte = (value: number) => Math.round(Math.min(1, Math.max(0, value)) * 255);
  // splat colors are display-referred: keep them as sRGB bytes
  const texel = [[0, 0], [1, 0], [0, 1]];

  for (let t = 0; t < triangles; t++) {
    const cx = (t % cellsPerRow) * 2;
    const cy = Math.floor(t / cellsPerRow) * 2;
    const corner: Vec3[] = [];
    for (let k = 0; k < 3; k++) {
      const v = mesh.indices[t * 3 + k];
      positions.set(mesh.positions.subarray(v * 3, v * 3 + 3), t * 9 + k * 3);
      normals.set(mesh.normals.subarray(v * 3, v * 3 + 3), t * 9 + k * 3);
      const rgb: Vec3 = [mesh.colors[v * 3], mesh.colors[v * 3 + 1], mesh.colors[v * 3 + 2]];
      corner.push(rgb);
      const px = cx + texel[k][0], py = cy + texel[k][1];
      const p = (py * size + px) * 4;
      pixels[p] = toByte(rgb[0]); pixels[p + 1] = toByte(rgb[1]); pixels[p + 2] = toByte(rgb[2]); pixels[p + 3] = 255;
      uvs[t * 6 + k * 2] = (px + 0.5) / size;
      uvs[t * 6 + k * 2 + 1] = (py + 0.5) / size;
    }
    // fourth texel completes the plane through the three corners
    const p = ((cy + 1) * size + cx + 1) * 4;
    for (let c = 0; c < 3; c++) pixels[p + c] = toByte(corner[1][c] + corner[2][c] - corner[0][c]);
    pixels[p + 3] = 255;
  }
  return { size, pixels, positions, normals, uvs };
}
