import assert from "node:assert/strict";
import test from "node:test";
import { bakeTextureAtlas, buildScanMesh, decimate, orbitCenter, subjectCylinder, type CameraModel, type ScanCamera, type SplatCloud, type Vec3 } from "../pages-src/scan-mesh";

const logit = (p: number) => Math.log(p / (1 - p));

// OpenCV-style world (y down), like splat.js: the subject sits at (0, 0, 2),
// a ball of radius 0.2 resting on a table at y = 0.2 (below it, since y is down).
const CENTER: Vec3 = [0, 0, 2];
const RADIUS = 0.2;
const TABLE_Y = 0.2;

function orbitCameras(count = 24): ScanCamera[] {
  return Array.from({ length: count }, (_, i) => {
    const angle = (i / count) * Math.PI * 2;
    const position: Vec3 = [CENTER[0] + 0.8 * Math.sin(angle), CENTER[1] - 0.35, CENTER[2] - 0.8 * Math.cos(angle)];
    const toward: Vec3 = [CENTER[0] - position[0], CENTER[1] - position[1], CENTER[2] - position[2]];
    const l = Math.hypot(...toward);
    return { position, forward: toward.map((v) => v / l) as Vec3, up: [0, -1, 0] };
  });
}

function sceneCloud(): SplatCloud {
  let seed = 42;
  const random = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  const rows: number[][] = [];
  const add = (p: Vec3, rgb: Vec3, logScale: number) =>
    rows.push([...p, logScale, logScale, logScale, 1, 0, 0, 0, ...rgb.map((c) => logit(Math.min(0.99, Math.max(0.01, c)))), logit(0.95), 0, 0]);
  // ball: red on top (−y), blue on the bottom
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < 6000; i++) {
    const y = 1 - (i / 5999) * 2;
    const r = Math.sqrt(1 - y * y);
    const p: Vec3 = [CENTER[0] + RADIUS * r * Math.cos(golden * i), CENTER[1] + RADIUS * y, CENTER[2] + RADIUS * r * Math.sin(golden * i)];
    add(p, y < 0 ? [0.9, 0.1, 0.1] : [0.1, 0.1, 0.9], Math.log(0.004));
  }
  // table plane under it
  for (let i = 0; i < 8000; i++) {
    const p: Vec3 = [CENTER[0] + (random() - 0.5) * 1.0, TABLE_Y, CENTER[2] + (random() - 0.5) * 1.0];
    add(p, [0.6, 0.5, 0.3], Math.log(0.01));
  }
  // floaters
  for (let i = 0; i < 30; i++) add([CENTER[0] + 0.35, CENTER[1] - 0.2 + i * 0.001, CENTER[2]], [1, 1, 1], Math.log(0.002));
  return { data: Float32Array.from(rows.flat()), n: rows.length, dc: "sigmoid" };
}

test("orbit center finds what the cameras look at", () => {
  const center = orbitCenter(orbitCameras())!;
  assert.ok(center);
  // all axes pass through CENTER
  center.forEach((value, k) => assert.ok(Math.abs(value - CENTER[k]) < 1e-6, `axis ${k}: ${value}`));
});

test("splat scene becomes an upright, ground-free, outward-facing mesh", () => {
  const mesh = buildScanMesh(sceneCloud(), orbitCameras(), { heightMeters: 0.4 });
  assert.ok(mesh.stats.groundRemoved, "table detected");
  assert.ok(mesh.stats.triangles > 1000, `triangles: ${mesh.stats.triangles}`);

  // ball diameter 0.4 → the width should match the height (ground not included)
  const [width, height, depth] = mesh.stats.sizeMeters;
  // a ball meets the table at a point, so its rebuilt base is a thin column the smoothing partly erodes
  assert.ok(Math.abs(height - 0.4) < 0.02, `height ${height}`);
  assert.ok(Math.abs(width - 0.4) < 0.06, `width ${width}`);
  assert.ok(Math.abs(depth - 0.4) < 0.06, `depth ${depth}`);

  // the top sits at the requested height above the ground, the base on it; normals point away from the ball center (y = 0.2)
  let minY = Infinity, maxY = -Infinity, outward = 0;
  const vertices = mesh.positions.length / 3;
  for (let v = 0; v < vertices; v++) {
    const p = [mesh.positions[v * 3], mesh.positions[v * 3 + 1] - 0.2, mesh.positions[v * 3 + 2]];
    minY = Math.min(minY, mesh.positions[v * 3 + 1]);
    maxY = Math.max(maxY, mesh.positions[v * 3 + 1]);
    if (p[0] * mesh.normals[v * 3] + p[1] * mesh.normals[v * 3 + 1] + p[2] * mesh.normals[v * 3 + 2] > 0) outward++;
  }
  assert.ok(Math.abs(maxY - 0.4) < 1e-6, `top ${maxY}`);
  assert.ok(Math.abs(minY) < 0.02, `base ${minY}`);
  assert.ok(outward / vertices > 0.97, `outward ${outward / vertices}`);

  // red on top, blue underneath (up is +Y in the output)
  let topRed = 0, top = 0, bottomBlue = 0, bottom = 0;
  for (let v = 0; v < vertices; v++) {
    const y = mesh.positions[v * 3 + 1];
    const [r, , b] = [mesh.colors[v * 3], mesh.colors[v * 3 + 1], mesh.colors[v * 3 + 2]];
    if (y > 0.32) { top++; if (r > b) topRed++; }
    if (y < 0.08) { bottom++; if (b > r) bottomBlue++; }
  }
  assert.ok(topRed / top > 0.9, `top red ${topRed}/${top}`);
  assert.ok(bottomBlue / bottom > 0.9, `bottom blue ${bottomBlue}/${bottom}`);
});

test("winding agrees with the normals", () => {
  const mesh = buildScanMesh(sceneCloud(), orbitCameras());
  let agree = 0;
  const triangles = mesh.indices.length / 3;
  for (let t = 0; t < triangles; t++) {
    const [a, b, c] = [mesh.indices[t * 3] * 3, mesh.indices[t * 3 + 1] * 3, mesh.indices[t * 3 + 2] * 3];
    const ab = [0, 1, 2].map((k) => mesh.positions[b + k] - mesh.positions[a + k]);
    const ac = [0, 1, 2].map((k) => mesh.positions[c + k] - mesh.positions[a + k]);
    const face = [ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2], ab[0] * ac[1] - ab[1] * ac[0]];
    if (face[0] * mesh.normals[a] + face[1] * mesh.normals[a + 1] + face[2] * mesh.normals[a + 2] > 0) agree++;
  }
  // computeNormals derives from winding, so this checks consistency of the triangle soup
  assert.ok(agree / triangles > 0.97);
});

function edgeUse(indices: Uint32Array) {
  const uses = new Map<string, number>();
  for (let t = 0; t < indices.length; t += 3) {
    for (let k = 0; k < 3; k++) {
      const a = indices[t + k], b = indices[t + (k + 1) % 3];
      const id = a < b ? `${a},${b}` : `${b},${a}`;
      uses.set(id, (uses.get(id) ?? 0) + 1);
    }
  }
  return [...uses.values()];
}

test("decimation keeps a closed, same-sized, outward-facing surface", () => {
  const full = buildScanMesh(sceneCloud(), orbitCameras(), { heightMeters: 0.4 });
  const histogram = (counts: number[]) => counts.reduce<Record<number, number>>((h, c) => ({ ...h, [c]: (h[c] ?? 0) + 1 }), {});
  assert.deepEqual(Object.keys(histogram(edgeUse(full.indices))), ["2"], `marching cubes edges ${JSON.stringify(histogram(edgeUse(full.indices)))}`);
  const target = Math.round(full.stats.triangles / 6);
  const mesh = buildScanMesh(sceneCloud(), orbitCameras(), { heightMeters: 0.4, targetTriangles: target });
  assert.ok(mesh.stats.triangles <= target + 2, `triangles ${mesh.stats.triangles} > ${target}`);
  assert.ok(mesh.stats.triangles > target * 0.9, `over-decimated to ${mesh.stats.triangles}`);
  // watertight: every edge shared by exactly two triangles
  assert.ok(edgeUse(mesh.indices).every((count) => count === 2), `edges after decimation ${JSON.stringify(histogram(edgeUse(mesh.indices)))}`);
  mesh.stats.sizeMeters.forEach((value, k) => assert.ok(Math.abs(value - full.stats.sizeMeters[k]) < 0.02, `axis ${k}: ${value} vs ${full.stats.sizeMeters[k]}`));
  let outward = 0;
  const vertices = mesh.positions.length / 3;
  for (let v = 0; v < vertices; v++) {
    const p = [mesh.positions[v * 3], mesh.positions[v * 3 + 1] - 0.2, mesh.positions[v * 3 + 2]];
    if (p[0] * mesh.normals[v * 3] + p[1] * mesh.normals[v * 3 + 1] + p[2] * mesh.normals[v * 3 + 2] > 0) outward++;
  }
  assert.ok(outward / vertices > 0.97, `outward ${outward / vertices}`);
});

test("decimate leaves a flat grid flat and stops at the target", () => {
  const side = 30;
  const positions: number[] = [];
  const indices: number[] = [];
  for (let y = 0; y <= side; y++) for (let x = 0; x <= side; x++) positions.push(x / side, y / side, 0);
  for (let y = 0; y < side; y++) for (let x = 0; x < side; x++) {
    const a = y * (side + 1) + x, b = a + 1, c = a + side + 1, d = c + 1;
    indices.push(a, b, d, a, d, c);
  }
  const out = decimate(new Float32Array(positions), new Uint32Array(indices), 200);
  assert.ok(out.indices.length / 3 <= 202);
  for (let v = 0; v < out.positions.length / 3; v++) assert.ok(Math.abs(out.positions[v * 3 + 2]) < 1e-9);
});

test("texture atlas keeps every triangle's samples inside its own half-cell", () => {
  const mesh = buildScanMesh(sceneCloud(), orbitCameras(), { targetTriangles: 3000 });
  const atlas = bakeTextureAtlas(mesh, { size: 512 });
  const triangles = mesh.indices.length / 3;
  assert.equal(atlas.positions.length, triangles * 9);
  const n = atlas.cell;
  assert.ok(n >= 4);
  for (let t = 0; t < triangles; t++) {
    const texels = [0, 1, 2].map((k) => [atlas.uvs[t * 6 + k * 2] * atlas.size - 0.5, atlas.uvs[t * 6 + k * 2 + 1] * atlas.size - 0.5]);
    const cells = texels.map(([x, y]) => `${Math.floor(x / n)},${Math.floor(y / n)}`);
    assert.ok(cells.every((cell) => cell === cells[0]), `triangle ${t} spans cells`);
    // bilinear footprint of the hypotenuse stays on this triangle's side of the gap
    const diagonals = texels.map(([x, y]) => (x % n) + (y % n));
    if (t % 2 === 0) assert.ok(Math.max(...diagonals) <= n - 2 + 1e-6, `lower triangle ${t} reaches ${Math.max(...diagonals)}`);
    else assert.ok(Math.min(...diagonals) >= n + 1 - 1e-6, `upper triangle ${t} reaches ${Math.min(...diagonals)}`);
  }
  // the red top really is red in the texture
  let red = 0, samples = 0;
  for (let t = 0; t < triangles; t++) {
    const y = (atlas.positions[t * 9 + 1] + atlas.positions[t * 9 + 4] + atlas.positions[t * 9 + 7]) / 3;
    if (y < 0.25) continue;
    const u = (atlas.uvs[t * 6] + atlas.uvs[t * 6 + 2] + atlas.uvs[t * 6 + 4]) / 3;
    const v = (atlas.uvs[t * 6 + 1] + atlas.uvs[t * 6 + 3] + atlas.uvs[t * 6 + 5]) / 3;
    const o = (Math.floor(v * atlas.size) * atlas.size + Math.floor(u * atlas.size)) * 4;
    samples++;
    if (atlas.pixels[o] > atlas.pixels[o + 2]) red++;
  }
  assert.ok(red / samples > 0.9, `red ${red}/${samples}`);
});

/** OpenCV pinhole models for orbitCameras(), as the solver would report them. */
function cameraModels(cameras: ScanCamera[]): CameraModel[] {
  return cameras.map(({ position, forward, up }) => {
    const z = forward;
    const down: Vec3 = [-up[0], -up[1], -up[2]];
    const along = down[0] * z[0] + down[1] * z[1] + down[2] * z[2];
    let y: Vec3 = [down[0] - along * z[0], down[1] - along * z[1], down[2] - along * z[2]];
    const ly = Math.hypot(...y);
    y = y.map((v) => v / ly) as Vec3;
    const x: Vec3 = [y[1] * z[2] - y[2] * z[1], y[2] * z[0] - y[0] * z[2], y[0] * z[1] - y[1] * z[0]];
    const R = [...x, ...y, ...z];
    const t = [0, 1, 2].map((r) => -(R[r * 3] * position[0] + R[r * 3 + 1] * position[1] + R[r * 3 + 2] * position[2]));
    return { R, t, fx: 400, fy: 400, cx: 320, cy: 240, width: 640, height: 480 };
  });
}

test("a broken density shell does not cave in when the cameras saw a closed surface", () => {
  // a strict density threshold (the Density slider at its top end) cuts the shell apart
  const inward = (mesh: ReturnType<typeof buildScanMesh>) => {
    // a hollow or pitted ball shows surface whose normals face its center
    let count = 0;
    const vertices = mesh.positions.length / 3;
    for (let v = 0; v < vertices; v++) {
      const p = [mesh.positions[v * 3], mesh.positions[v * 3 + 1] - 0.2, mesh.positions[v * 3 + 2]];
      if (p[0] * mesh.normals[v * 3] + p[1] * mesh.normals[v * 3 + 1] + p[2] * mesh.normals[v * 3 + 2] < 0) count++;
    }
    return count / vertices;
  };
  const options = { heightMeters: 0.4, density: 0.95 };
  const shellOnly = buildScanMesh(sceneCloud(), orbitCameras(), options);
  const carved = buildScanMesh(sceneCloud(), orbitCameras(), { ...options, views: cameraModels(orbitCameras()) });
  assert.ok(inward(shellOnly) > 0.05, `the strict threshold should break the shell (${inward(shellOnly)})`);
  assert.ok(inward(carved) < 0.02, `carving should keep the ball solid (${inward(carved)})`);
});

test("the training region hugs the object's sparse points, not the camera ring", () => {
  // sparse points as a solver leaves them: some on the ball, many on the table around it
  const cloud = sceneCloud();
  const points: Vec3[] = [];
  for (let i = 0; i < cloud.n; i += 7) points.push([cloud.data[i * 16], cloud.data[i * 16 + 1], cloud.data[i * 16 + 2]]);
  const cameras = orbitCameras();
  const region = subjectCylinder(points, cameras);
  // local = rotation · (world − center); the cameras' ring is 0.8 m out
  const local = (p: Vec3) => [0, 1, 2].map((r) => region.rotation[r * 3] * (p[0] - region.center[0]) + region.rotation[r * 3 + 1] * (p[1] - region.center[1]) + region.rotation[r * 3 + 2] * (p[2] - region.center[2]));
  assert.ok(region.radius > RADIUS * 1.15 && region.radius < 0.36, `radius ${region.radius}`);
  // every ball point is inside, with room to spare
  for (let i = 0; i < 6000; i += 13) {
    const [x, y, z] = local([cloud.data[i * 16], cloud.data[i * 16 + 1], cloud.data[i * 16 + 2]]);
    assert.ok(Math.hypot(x, z) < region.radius && y > region.bottom && y < region.top, `ball point ${i} outside`);
  }
});
