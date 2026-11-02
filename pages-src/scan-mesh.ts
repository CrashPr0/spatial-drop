// Turns a trained Gaussian splat into a watertight, colored triangle mesh that
// every AR viewer understands. Everything here is plain typed-array math so it
// runs on the phone that made the scan and can be unit tested in Node.

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
  /** decimate the surface down to about this many triangles */
  targetTriangles?: number;
}

export interface ScanMesh {
  positions: Float32Array;
  normals: Float32Array;
  colors: Float32Array;
  indices: Uint32Array;
  /** splat color at a point on (or near) the output surface, in output meters */
  sampleColor: (point: Vec3, out: Vec3) => Vec3;
  /** output meters back to the reconstruction's world frame (where the cameras live) */
  toWorld: (point: Vec3) => Vec3;
  directionToWorld: (direction: Vec3) => Vec3;
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

/** transpose(m) · p, the inverse of a rotation */
const applyTransposed = (m: number[], p: Vec3): Vec3 => [
  m[0] * p[0] + m[3] * p[1] + m[6] * p[2],
  m[1] * p[0] + m[4] * p[1] + m[7] * p[2],
  m[2] * p[0] + m[5] * p[1] + m[8] * p[2],
];

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
  let level = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  if (options.removeGround !== false) {
    const solid = splats.filter((splat) => splat.alpha > 0.5).map((splat) => splat.position);
    // a trained splat surface is a few percent of the subject radius thick
    const ground = findGround(solid, 0.04 * radius);
    if (ground) {
      // level the subject on the ground plane, then cut the plane away
      level = rotationBetween(ground.normal, [0, 1, 0]);
      for (const splat of splats) {
        splat.position = apply(level, splat.position);
        splat.axes = splat.axes.map((axis) => apply(level, axis)) as [Vec3, Vec3, Vec3];
      }
      floor = ground.offset + 0.06 * radius;
      groundRemoved = true;
    }
  }
  const subject = splats.filter((splat) => splat.position[1] > floor);
  // local = level · basis · (world − center)
  return { splats: subject, radius, floor, groundRemoved, transform: { center: center as Vec3, basis, level } };
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

// Kuhn split of a cube into six tetrahedra along its main diagonal: every
// face diagonal agrees with the neighboring cube's, so the surface has no
// cracks (classic marching cubes tables leave holes in ambiguous cases).
// Corners are bit masks: x = 1, y = 2, z = 4.
const KUHN_TETS = [[1, 2, 4], [1, 4, 2], [2, 1, 4], [2, 4, 1], [4, 1, 2], [4, 2, 1]].map(([a, b]) => [0, a, a | b, 7]);

/** Marching tetrahedra over a scalar field; triangles face from high values toward low ones. */
function marchingTetrahedra(field: Float32Array, grid: Grid, iso: number) {
  const [nx, ny, nz] = grid.dims;
  // edges join a corner to a superset corner: key by the lower corner and the direction bits
  const edgeVertex = new Map<number, number>();
  const positions: number[] = [];
  const indices: number[] = [];
  const cornerValue = new Float64Array(8);
  const corner = (x: number, y: number, z: number, c: number) => index(grid, x + (c & 1), y + ((c >> 1) & 1), z + ((c >> 2) & 1));

  const vertexOn = (x: number, y: number, z: number, a: number, b: number) => {
    const key = corner(x, y, z, a) * 8 + (a ^ b);
    let id = edgeVertex.get(key);
    if (id === undefined) {
      const fa = cornerValue[a], fb = cornerValue[b];
      const t = (iso - fa) / (fb - fa);
      id = positions.length / 3;
      edgeVertex.set(key, id);
      positions.push(
        grid.origin[0] + (x + (a & 1) + t * ((b & 1) - (a & 1))) * grid.cell,
        grid.origin[1] + (y + ((a >> 1) & 1) + t * (((b >> 1) & 1) - ((a >> 1) & 1))) * grid.cell,
        grid.origin[2] + (z + ((a >> 2) & 1) + t * (((b >> 2) & 1) - ((a >> 2) & 1))) * grid.cell,
      );
    }
    return id;
  };
  const cornerPosition = (c: number): Vec3 => [c & 1, (c >> 1) & 1, (c >> 2) & 1];
  // emit a triangle facing away from the inside corners
  const emit = (i0: number, i1: number, i2: number, inside: number[], outside: number[]) => {
    const p = (i: number): Vec3 => [positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]];
    const a = p(i0), b = p(i1), c = p(i2);
    const n = cross(sub(b, a), sub(c, a));
    const mean = (corners: number[]) => corners.reduce<Vec3>((sum, k) => {
      const q = cornerPosition(k);
      return [sum[0] + q[0] / corners.length, sum[1] + q[1] / corners.length, sum[2] + q[2] / corners.length];
    }, [0, 0, 0]);
    const outward = sub(mean(outside), mean(inside));
    if (dot(n, outward) >= 0) indices.push(i0, i1, i2);
    else indices.push(i0, i2, i1);
  };

  for (let z = 0; z < nz - 1; z++) for (let y = 0; y < ny - 1; y++) for (let x = 0; x < nx - 1; x++) {
    let any = 0, all = 1;
    for (let c = 0; c < 8; c++) {
      cornerValue[c] = field[corner(x, y, z, c)];
      if (cornerValue[c] > iso) any = 1; else all = 0;
    }
    if (!any || all) continue;
    for (const tet of KUHN_TETS) {
      const inside = tet.filter((c) => cornerValue[c] > iso);
      if (inside.length === 0 || inside.length === 4) continue;
      const outside = tet.filter((c) => cornerValue[c] <= iso);
      const edge = (a: number, b: number) => vertexOn(x, y, z, Math.min(a, b), Math.max(a, b));
      if (inside.length === 1 || inside.length === 3) {
        const [lone, others] = inside.length === 1 ? [inside[0], outside] : [outside[0], inside];
        emit(edge(lone, others[0]), edge(lone, others[1]), edge(lone, others[2]), inside, outside);
      } else {
        const [a, b] = inside, [c, d] = outside;
        const ac = edge(a, c), ad = edge(a, d), bd = edge(b, d), bc = edge(b, c);
        emit(ac, ad, bd, inside, outside);
        emit(ac, bd, bc, inside, outside);
      }
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
  const { splats, groundRemoved, floor, transform } = prepareSubject(cloud, cameras, options);
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
  const cell = extent / (resolution - 10);
  const grid: Grid = {
    cell,
    origin: [lo[0] - 5 * cell, lo[1] - 5 * cell, lo[2] - 5 * cell],
    dims: [0, 1, 2].map((k) => Math.ceil((hi[k] - lo[k]) / cell) + 11) as Vec3,
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

  // keep a clear margin so the blurred field is zero on the border and the surface closes
  const field = new Float32Array(total);
  for (let z = 4; z < nz - 4; z++) for (let y = 4; y < ny - 4; y++) for (let x = 4; x < nx - 4; x++) {
    const i = index(grid, x, y, z);
    field[i] = solid[i];
  }
  const smoothField = blur(field, grid.dims);
  // just off 0.5: the blurred 0/1 field hits 0.5 exactly, which would put vertices on grid corners
  let { positions, indices } = marchingTetrahedra(smoothField, grid, 0.5 - 1e-4);
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

  const voxelColor = (p: Vec3, out: Vec3) => {
    const g: Vec3 = [(p[0] - grid.origin[0]) / cell, (p[1] - grid.origin[1]) / cell, (p[2] - grid.origin[2]) / cell];
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
    if (w > 0) { out[0] = r / w; out[1] = gr / w; out[2] = b / w; } else { out[0] = out[1] = out[2] = 0.6; }
    return out;
  };
  const splatColor = splatColorSampler(splats, cell, voxelColor);

  if (options.targetTriangles && indices.length / 3 > options.targetTriangles) {
    ({ positions, indices } = decimate(positions, indices, options.targetTriangles));
  }

  const vertexCount = positions.length / 3;
  const colors = new Float32Array(vertexCount * 3);
  const sampled: Vec3 = [0, 0, 0];
  for (let v = 0; v < vertexCount; v++) {
    splatColor([positions[v * 3], positions[v * 3 + 1], positions[v * 3 + 2]], sampled);
    colors.set(sampled, v * 3);
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
  const local: Vec3 = [0, 0, 0];

  return {
    positions,
    normals: computeNormals(positions, indices),
    colors,
    indices,
    sampleColor(p, out) {
      for (let k = 0; k < 3; k++) local[k] = p[k] / metersPerUnit + middle[k];
      return splatColor(local, out);
    },
    toWorld(p) {
      const back = applyTransposed(transform.basis, applyTransposed(transform.level, [
        p[0] / metersPerUnit + middle[0], p[1] / metersPerUnit + middle[1], p[2] / metersPerUnit + middle[2],
      ]));
      return [back[0] + transform.center[0], back[1] + transform.center[1], back[2] + transform.center[2]];
    },
    directionToWorld: (d) => applyTransposed(transform.basis, applyTransposed(transform.level, d)),
    stats: {
      splatsUsed: used,
      groundRemoved,
      voxels: grid.dims,
      triangles: indices.length / 3,
      sizeMeters: [(max[0] - min[0]) * metersPerUnit, height * metersPerUnit, (max[2] - min[2]) * metersPerUnit],
    },
  };
}

/** Color at a surface point straight from the nearby splats (sharper than the
 *  voxel grid, which only sees one color per voxel); the voxel grid fills in
 *  where no splat reaches. */
function splatColorSampler(splats: PreparedSplat[], cell: number, fallback: (p: Vec3, out: Vec3) => Vec3) {
  const buckets = new Map<number, number[]>();
  const key = (x: number, y: number, z: number) => ((x + 1024) * 2048 + (y + 1024)) * 2048 + (z + 1024);
  splats.forEach((splat, i) => {
    const k = key(Math.floor(splat.position[0] / cell), Math.floor(splat.position[1] / cell), Math.floor(splat.position[2] / cell));
    const bucket = buckets.get(k);
    if (bucket) bucket.push(i); else buckets.set(k, [i]);
  });
  const minSigma = 0.3 * cell;
  return (p: Vec3, out: Vec3) => {
    const bx = Math.floor(p[0] / cell), by = Math.floor(p[1] / cell), bz = Math.floor(p[2] / cell);
    let r = 0, g = 0, b = 0, w = 0;
    for (let dz = -1; dz <= 1; dz++) for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const bucket = buckets.get(key(bx + dx, by + dy, bz + dz));
      if (!bucket) continue;
      for (const i of bucket) {
        const splat = splats[i];
        const d: Vec3 = [p[0] - splat.position[0], p[1] - splat.position[1], p[2] - splat.position[2]];
        let m = 0;
        for (let k = 0; k < 3; k++) {
          const projected = dot(splat.axes[k], d) / Math.max(splat.sigmas[k], minSigma);
          m += projected * projected;
        }
        if (m > 16) continue;
        const weight = splat.alpha * Math.exp(-0.5 * m);
        r += weight * splat.color[0]; g += weight * splat.color[1]; b += weight * splat.color[2]; w += weight;
      }
    }
    if (w < 1e-3) return fallback(p, out);
    out[0] = r / w; out[1] = g / w; out[2] = b / w;
    return out;
  };
}

/** Quadric error edge collapse (Garland & Heckbert): fewer triangles where
 *  the surface is flat, detail kept where it bends. Collapses that would flip
 *  a neighboring face or pinch the surface are refused. */
export function decimate(positionsIn: Float32Array, indicesIn: Uint32Array, targetTriangles: number) {
  const vertexCount = positionsIn.length / 3;
  const pos = Float64Array.from(positionsIn);
  const faces = Int32Array.from(indicesIn);
  const faceCount = faces.length / 3;
  const faceAlive = new Uint8Array(faceCount).fill(1);
  let aliveFaces = faceCount;
  const quadric = new Float64Array(vertexCount * 10);
  const vertexFaces: number[][] = Array.from({ length: vertexCount }, () => []);
  const vertexAlive = new Uint8Array(vertexCount).fill(1);
  const version = new Uint32Array(vertexCount);

  const faceNormal = (a: number, b: number, c: number, out: number[]) => {
    const ux = pos[b * 3] - pos[a * 3], uy = pos[b * 3 + 1] - pos[a * 3 + 1], uz = pos[b * 3 + 2] - pos[a * 3 + 2];
    const vx = pos[c * 3] - pos[a * 3], vy = pos[c * 3 + 1] - pos[a * 3 + 1], vz = pos[c * 3 + 2] - pos[a * 3 + 2];
    out[0] = uy * vz - uz * vy; out[1] = uz * vx - ux * vz; out[2] = ux * vy - uy * vx;
    return out;
  };
  const normal = [0, 0, 0];
  for (let f = 0; f < faceCount; f++) {
    const a = faces[f * 3], b = faces[f * 3 + 1], c = faces[f * 3 + 2];
    faceNormal(a, b, c, normal);
    const area = Math.hypot(normal[0], normal[1], normal[2]);
    if (area > 1e-20) {
      const nx = normal[0] / area, ny = normal[1] / area, nz = normal[2] / area;
      const d = -(nx * pos[a * 3] + ny * pos[a * 3 + 1] + nz * pos[a * 3 + 2]);
      const plane = [nx * nx, nx * ny, nx * nz, nx * d, ny * ny, ny * nz, ny * d, nz * nz, nz * d, d * d];
      for (const v of [a, b, c]) for (let k = 0; k < 10; k++) quadric[v * 10 + k] += area * plane[k];
    }
    vertexFaces[a].push(f); vertexFaces[b].push(f); vertexFaces[c].push(f);
  }

  const q = new Float64Array(10);
  const error = (x: number, y: number, z: number) =>
    q[0] * x * x + 2 * q[1] * x * y + 2 * q[2] * x * z + 2 * q[3] * x + q[4] * y * y + 2 * q[5] * y * z + 2 * q[6] * y + q[7] * z * z + 2 * q[8] * z + q[9];

  // binary min-heap of candidate collapses, invalidated lazily by vertex versions
  const heapCost: number[] = [], heapU: number[] = [], heapV: number[] = [], heapVu: number[] = [], heapVv: number[] = [];
  const heapX: number[] = [], heapY: number[] = [], heapZ: number[] = [];
  const swap = (i: number, j: number) => {
    for (const array of [heapCost, heapU, heapV, heapVu, heapVv, heapX, heapY, heapZ]) [array[i], array[j]] = [array[j], array[i]];
  };
  const push = (u: number, v: number) => {
    for (let k = 0; k < 10; k++) q[k] = quadric[u * 10 + k] + quadric[v * 10 + k];
    const mx = (pos[u * 3] + pos[v * 3]) / 2, my = (pos[u * 3 + 1] + pos[v * 3 + 1]) / 2, mz = (pos[u * 3 + 2] + pos[v * 3 + 2]) / 2;
    const candidates = [[pos[u * 3], pos[u * 3 + 1], pos[u * 3 + 2]], [pos[v * 3], pos[v * 3 + 1], pos[v * 3 + 2]], [mx, my, mz]];
    // the quadric's own minimum, when it is well defined and near the edge
    const det = q[0] * (q[4] * q[7] - q[5] * q[5]) - q[1] * (q[1] * q[7] - q[5] * q[2]) + q[2] * (q[1] * q[5] - q[4] * q[2]);
    const edge = Math.hypot(pos[u * 3] - pos[v * 3], pos[u * 3 + 1] - pos[v * 3 + 1], pos[u * 3 + 2] - pos[v * 3 + 2]);
    if (Math.abs(det) > 1e-12 * Math.max(1e-30, (q[0] + q[4] + q[7]) ** 3)) {
      const bx = -q[3], by = -q[6], bz = -q[8];
      const x = (bx * (q[4] * q[7] - q[5] * q[5]) - q[1] * (by * q[7] - q[5] * bz) + q[2] * (by * q[5] - q[4] * bz)) / det;
      const y = (q[0] * (by * q[7] - q[5] * bz) - bx * (q[1] * q[7] - q[5] * q[2]) + q[2] * (q[1] * bz - by * q[2])) / det;
      const z = (q[0] * (q[4] * bz - by * q[5]) - q[1] * (q[1] * bz - by * q[2]) + bx * (q[1] * q[5] - q[4] * q[2])) / det;
      if (Math.hypot(x - mx, y - my, z - mz) < edge) candidates.push([x, y, z]);
    }
    let best = candidates[0], bestCost = Infinity;
    for (const candidate of candidates) {
      const cost = error(candidate[0], candidate[1], candidate[2]);
      if (cost < bestCost) { bestCost = cost; best = candidate; }
    }
    let i = heapCost.length;
    heapCost.push(Math.max(0, bestCost)); heapU.push(u); heapV.push(v); heapVu.push(version[u]); heapVv.push(version[v]);
    heapX.push(best[0]); heapY.push(best[1]); heapZ.push(best[2]);
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (heapCost[parent] <= heapCost[i]) break;
      swap(i, parent);
      i = parent;
    }
  };
  const pop = () => {
    const last = heapCost.length - 1;
    swap(0, last);
    const top = { u: heapU[last], v: heapV[last], vu: heapVu[last], vv: heapVv[last], x: heapX[last], y: heapY[last], z: heapZ[last] };
    for (const array of [heapCost, heapU, heapV, heapVu, heapVv, heapX, heapY, heapZ]) array.pop();
    let i = 0;
    for (;;) {
      const l = 2 * i + 1, r = l + 1;
      let smallest = i;
      if (l < heapCost.length && heapCost[l] < heapCost[smallest]) smallest = l;
      if (r < heapCost.length && heapCost[r] < heapCost[smallest]) smallest = r;
      if (smallest === i) break;
      swap(i, smallest);
      i = smallest;
    }
    return top;
  };

  const neighbors = (v: number) => {
    const set = new Set<number>();
    for (const f of vertexFaces[v]) {
      if (!faceAlive[f]) continue;
      for (let k = 0; k < 3; k++) if (faces[f * 3 + k] !== v) set.add(faces[f * 3 + k]);
    }
    return set;
  };
  const seen = new Set<number>();
  for (let f = 0; f < faceCount; f++) {
    for (let k = 0; k < 3; k++) {
      const a = faces[f * 3 + k], b = faces[f * 3 + (k + 1) % 3];
      const u = Math.min(a, b), v = Math.max(a, b);
      const id = u * vertexCount + v;
      if (seen.has(id)) continue;
      seen.add(id);
      push(u, v);
    }
  }
  seen.clear();

  const before = [0, 0, 0], after = [0, 0, 0];
  const flips = (moving: number, other: number, x: number, y: number, z: number) => {
    const ox = pos[moving * 3], oy = pos[moving * 3 + 1], oz = pos[moving * 3 + 2];
    for (const f of vertexFaces[moving]) {
      if (!faceAlive[f]) continue;
      const a = faces[f * 3], b = faces[f * 3 + 1], c = faces[f * 3 + 2];
      if (a === other || b === other || c === other) continue; // removed by the collapse
      faceNormal(a, b, c, before);
      pos[moving * 3] = x; pos[moving * 3 + 1] = y; pos[moving * 3 + 2] = z;
      faceNormal(a, b, c, after);
      pos[moving * 3] = ox; pos[moving * 3 + 1] = oy; pos[moving * 3 + 2] = oz;
      const lb = Math.hypot(before[0], before[1], before[2]), la = Math.hypot(after[0], after[1], after[2]);
      if (la < 1e-20 || (before[0] * after[0] + before[1] * after[1] + before[2] * after[2]) < 0.3 * lb * la) return true;
    }
    return false;
  };

  while (aliveFaces > targetTriangles && heapCost.length) {
    const { u, v, vu, vv, x, y, z } = pop();
    if (!vertexAlive[u] || !vertexAlive[v] || version[u] !== vu || version[v] !== vv) continue;
    // link condition: exactly two shared neighbors keeps the surface a manifold
    const nu = neighbors(u), nv = neighbors(v);
    let shared = 0;
    for (const w of nu) if (nv.has(w)) shared++;
    if (shared !== 2) continue;
    if (flips(u, v, x, y, z) || flips(v, u, x, y, z)) continue;

    pos[u * 3] = x; pos[u * 3 + 1] = y; pos[u * 3 + 2] = z;
    for (let k = 0; k < 10; k++) quadric[u * 10 + k] += quadric[v * 10 + k];
    for (const f of vertexFaces[v]) {
      if (!faceAlive[f]) continue;
      const a = faces[f * 3], b = faces[f * 3 + 1], c = faces[f * 3 + 2];
      if (a === u || b === u || c === u) { faceAlive[f] = 0; aliveFaces--; continue; }
      for (let k = 0; k < 3; k++) if (faces[f * 3 + k] === v) faces[f * 3 + k] = u;
      vertexFaces[u].push(f);
    }
    vertexFaces[u] = vertexFaces[u].filter((f) => faceAlive[f]);
    vertexAlive[v] = 0;
    vertexFaces[v] = [];
    version[u]++;
    for (const w of neighbors(u)) push(u, w);
  }

  const remap = new Int32Array(vertexCount).fill(-1);
  const outPositions: number[] = [];
  const outIndices: number[] = [];
  for (let f = 0; f < faceCount; f++) {
    if (!faceAlive[f]) continue;
    for (let k = 0; k < 3; k++) {
      const v = faces[f * 3 + k];
      if (remap[v] < 0) { remap[v] = outPositions.length / 3; outPositions.push(pos[v * 3], pos[v * 3 + 1], pos[v * 3 + 2]); }
      outIndices.push(remap[v]);
    }
  }
  return { positions: new Float32Array(outPositions), indices: new Uint32Array(outIndices) };
}

/** Bake the surface color into a texture atlas. Every pair of triangles shares
 *  an N x N texel cell, split along the diagonal with a one-texel gap, and each
 *  texel is colored from the splats at the 3D point it maps to; texture detail
 *  no longer depends on how many triangles survived decimation. Bilinear
 *  samples inside a triangle only ever touch its own texels. Apple Quick Look
 *  ignores vertex colors on textured materials; a texture survives every
 *  converter. */
export function bakeTextureAtlas(mesh: ScanMesh, options: { size?: number; maxSize?: number } = {}) {
  const triangles = mesh.indices.length / 3;
  const pairs = Math.ceil(triangles / 2);
  let size = options.size ?? 1024;
  const maxSize = options.maxSize ?? 4096;
  let cellsPerRow = Math.ceil(Math.sqrt(pairs));
  let n = Math.floor(size / cellsPerRow);
  while (n < 4 && size < maxSize) {
    size *= 2;
    n = Math.floor(size / cellsPerRow);
  }
  if (n < 4) throw new Error("Too many triangles for the texture atlas. Choose a lower detail level.");
  n = Math.min(n, 24);
  cellsPerRow = Math.floor(size / n);

  // walk triangles and cells in Z-order so atlas neighbors are surface
  // neighbors: smooth texture that compresses well, and JPEG blocks that
  // straddle two cells blend similar colors
  const order = mortonOrder(triangles, (t, k) => {
    let sum = 0;
    for (let c = 0; c < 3; c++) sum += mesh.positions[mesh.indices[t * 3 + c] * 3 + k];
    return sum / 3;
  });
  const cellSlots: [number, number][] = [];
  for (let m = 0; cellSlots.length < pairs; m++) {
    let x = 0, y = 0;
    for (let bit = 0; bit < 16; bit++) {
      x |= ((m >> (2 * bit)) & 1) << bit;
      y |= ((m >> (2 * bit + 1)) & 1) << bit;
    }
    if (x < cellsPerRow && y < cellsPerRow) cellSlots.push([x, y]);
  }

  const pixels = new Uint8ClampedArray(size * size * 4);
  // the surface point and normal behind every texel, for photo projection
  const texelPoints = new Float32Array(size * size * 3);
  const texelNormals = new Float32Array(size * size * 3);
  const texelUsed = new Uint8Array(size * size);
  const indices = new Uint32Array(triangles * 3);
  const positions = new Float32Array(triangles * 9);
  const normals = new Float32Array(triangles * 9);
  const uvs = new Float32Array(triangles * 6);
  const toByte = (value: number) => Math.round(Math.min(1, Math.max(0, value)) * 255);
  const corner = (t: number, k: number): Vec3 => {
    const v = mesh.indices[order[t] * 3 + k];
    return [mesh.positions[v * 3], mesh.positions[v * 3 + 1], mesh.positions[v * 3 + 2]];
  };
  // texel-center coordinates of each half's corners inside a cell, as (column, row)
  const lower = [[0, 0], [n - 2, 0], [0, n - 2]];
  const upper = [[n - 1, n - 1], [n - 1, 2], [2, n - 1]];
  const point: Vec3 = [0, 0, 0];
  const rgb: Vec3 = [0, 0, 0];
  const cornerNormal = (t: number, k: number): Vec3 => {
    const v = mesh.indices[order[t] * 3 + k];
    return [mesh.normals[v * 3], mesh.normals[v * 3 + 1], mesh.normals[v * 3 + 2]];
  };

  for (let pair = 0; pair < pairs; pair++) {
    const ox = cellSlots[pair][0] * n;
    const oy = cellSlots[pair][1] * n;
    for (let half = 0; half < 2; half++) {
      const t = pair * 2 + half;
      if (t >= triangles) break;
      const layout = half ? upper : lower;
      for (let k = 0; k < 3; k++) {
        const v = mesh.indices[order[t] * 3 + k];
        indices[t * 3 + k] = v;
        positions.set(mesh.positions.subarray(v * 3, v * 3 + 3), t * 9 + k * 3);
        normals.set(mesh.normals.subarray(v * 3, v * 3 + 3), t * 9 + k * 3);
        uvs[t * 6 + k * 2] = (ox + layout[k][0] + 0.5) / size;
        uvs[t * 6 + k * 2 + 1] = (oy + layout[k][1] + 0.5) / size;
      }
      const p0 = corner(t, 0), p1 = corner(t, 1), p2 = corner(t, 2);
      const n0 = cornerNormal(t, 0), n1 = cornerNormal(t, 1), n2 = cornerNormal(t, 2);
      const legs = half ? n - 3 : n - 2;
      for (let row = 0; row < n; row++) {
        for (let column = 0; column < n; column++) {
          const diagonal = column + row;
          if (half ? diagonal < n : diagonal > n - 1) continue;
          // affine map from texel to triangle (extrapolates into the gap texels)
          const s = half ? (n - 1 - row) / legs : column / legs;
          const r = half ? (n - 1 - column) / legs : row / legs;
          for (let k = 0; k < 3; k++) point[k] = p0[k] + s * (p1[k] - p0[k]) + r * (p2[k] - p0[k]);
          mesh.sampleColor(point, rgb);
          const texel = (oy + row) * size + ox + column;
          texelUsed[texel] = 1;
          const w0 = 1 - s - r;
          const nx = w0 * n0[0] + s * n1[0] + r * n2[0];
          const ny = w0 * n0[1] + s * n1[1] + r * n2[1];
          const nz = w0 * n0[2] + s * n1[2] + r * n2[2];
          const nl = Math.hypot(nx, ny, nz) || 1;
          texelPoints[texel * 3] = point[0]; texelPoints[texel * 3 + 1] = point[1]; texelPoints[texel * 3 + 2] = point[2];
          texelNormals[texel * 3] = nx / nl; texelNormals[texel * 3 + 1] = ny / nl; texelNormals[texel * 3 + 2] = nz / nl;
          const o = texel * 4;
          pixels[o] = toByte(rgb[0]); pixels[o + 1] = toByte(rgb[1]); pixels[o + 2] = toByte(rgb[2]); pixels[o + 3] = 255;
        }
      }
    }
  }
  // `indices` lists the source mesh's triangles in atlas order; `uvs` follow it
  return { size, cell: n, pixels, indices, positions, normals, uvs, texelPoints, texelNormals, texelUsed };
}

/** Indices 0..count-1 sorted along a 3D Z-order curve of the given coordinates. */
function mortonOrder(count: number, coordinate: (i: number, axis: number) => number) {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < count; i++) for (let k = 0; k < 3; k++) {
    const value = coordinate(i, k);
    lo[k] = Math.min(lo[k], value);
    hi[k] = Math.max(hi[k], value);
  }
  const spread = (value: number) => {
    let x = value & 0x3ff;
    x = (x | (x << 16)) & 0x030000ff;
    x = (x | (x << 8)) & 0x0300f00f;
    x = (x | (x << 4)) & 0x030c30c3;
    x = (x | (x << 2)) & 0x09249249;
    return x;
  };
  const codes = new Float64Array(count);
  for (let i = 0; i < count; i++) {
    let code = 0;
    for (let k = 0; k < 3; k++) {
      const q = Math.floor(((coordinate(i, k) - lo[k]) / ((hi[k] - lo[k]) || 1)) * 1023);
      code += spread(q) * 2 ** k;
    }
    codes[i] = code;
  }
  return Array.from({ length: count }, (_, i) => i).sort((a, b) => codes[a] - codes[b]);
}
