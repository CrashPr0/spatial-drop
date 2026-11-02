import assert from "node:assert/strict";
import test from "node:test";
import { bakeColorAtlas, buildScanMesh, orbitCenter, type ScanCamera, type SplatCloud, type Vec3 } from "../pages-src/scan-mesh";

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
    const p: Vec3 = [CENTER[0] + (Math.random() - 0.5) * 1.0, TABLE_Y, CENTER[2] + (Math.random() - 0.5) * 1.0];
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
  assert.ok(Math.abs(height - 0.4) < 1e-6);
  assert.ok(Math.abs(width - 0.4) < 0.06, `width ${width}`);
  assert.ok(Math.abs(depth - 0.4) < 0.06, `depth ${depth}`);

  // resting on y = 0, normals point away from the ball center (y = 0.2)
  let minY = Infinity, outward = 0;
  const vertices = mesh.positions.length / 3;
  for (let v = 0; v < vertices; v++) {
    const p = [mesh.positions[v * 3], mesh.positions[v * 3 + 1] - 0.2, mesh.positions[v * 3 + 2]];
    minY = Math.min(minY, mesh.positions[v * 3 + 1]);
    if (p[0] * mesh.normals[v * 3] + p[1] * mesh.normals[v * 3 + 1] + p[2] * mesh.normals[v * 3 + 2] > 0) outward++;
  }
  assert.ok(Math.abs(minY) < 1e-6);
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

test("color atlas keeps every triangle inside its own texels", () => {
  const mesh = buildScanMesh(sceneCloud(), orbitCameras());
  const atlas = bakeColorAtlas(mesh);
  const triangles = mesh.indices.length / 3;
  assert.equal(atlas.positions.length, triangles * 9);
  assert.ok(atlas.size >= 2 * Math.ceil(Math.sqrt(triangles)));
  for (let t = 0; t < triangles; t += 97) {
    const us = [atlas.uvs[t * 6], atlas.uvs[t * 6 + 2], atlas.uvs[t * 6 + 4]].map((u) => Math.floor(u * atlas.size / 2));
    const vs = [atlas.uvs[t * 6 + 1], atlas.uvs[t * 6 + 3], atlas.uvs[t * 6 + 5]].map((v) => Math.floor(v * atlas.size / 2));
    assert.ok(us.every((u) => u === us[0]) && vs.every((v) => v === vs[0]), `triangle ${t} spans cells`);
  }
});
