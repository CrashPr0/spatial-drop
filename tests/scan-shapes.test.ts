import assert from "node:assert/strict";
import test from "node:test";
import { dimensionedCylinder, chooseShape, fitBox, fitRevolve, mightBeShape, shapeMesh, SHINY, type SurfaceSample } from "../pages-src/scan-shapes";

type Point = { p: [number, number, number]; n: [number, number, number] };

let seed = 7;
const random = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
const sampleOf = (points: Point[]): SurfaceSample => ({
  positions: Float32Array.from(points.flatMap(({ p }) => p)),
  normals: Float32Array.from(points.flatMap(({ n }) => n)),
  bottom: 0,
});

/** Points on an upright solid of revolution: radius(y, angle) gives each side point's distance from the axis. */
function revolved(height: number, radius: (y: number, angle: number) => number, count = 6000, at: [number, number] = [0.02, -0.01]) {
  const points: Point[] = [];
  for (let i = 0; i < count; i++) {
    const a = random() * 2 * Math.PI, y = random() * height;
    const r = radius(y, a);
    points.push({ p: [at[0] + r * Math.cos(a), y, at[1] + r * Math.sin(a)], n: [Math.cos(a), 0, Math.sin(a)] });
  }
  for (let i = 0; i < count / 6; i++) {
    const a = random() * 2 * Math.PI, d = Math.sqrt(random()) * radius(height, a);
    points.push({ p: [at[0] + d * Math.cos(a), height, at[1] + d * Math.sin(a)], n: [0, 1, 0] });
  }
  return points;
}

/** Points on an upright box turned `angle` about +Y. */
function boxed(size: [number, number, number], angle: number, count = 6000) {
  const [w, h, d] = size;
  const c = Math.cos(angle), s = Math.sin(angle);
  const turn = (u: number, v: number): [number, number] => [c * u - s * v, s * u + c * v];
  const points: Point[] = [];
  for (let i = 0; i < count; i++) {
    const y = random() * h, side = Math.floor(random() * 4), k = random() - 0.5;
    const [u, v, nu, nv] = side === 0 ? [w / 2, k * d, 1, 0] : side === 1 ? [-w / 2, k * d, -1, 0] : side === 2 ? [k * w, d / 2, 0, 1] : [k * w, -d / 2, 0, -1];
    const [x, z] = turn(u, v), [nx, nz] = turn(nu, nv);
    points.push({ p: [x, y, z], n: [nx, 0, nz] });
  }
  return points;
}

const noisy = (r: number, amount = 0.01) => r * (1 + amount * (random() * 2 - 1));
// what reflections do to a shiny can's scan: most of the surface pushed in, by up
// to 30% of the radius. Like the real can (2.93 cm median, 3.28 cm 75th
// percentile for 3.3 cm), the median falls 8% short and the outer part holds
const dented = (r: number) => (y: number, a: number) =>
  r * (1 - 0.3 * Math.max(0, 0.35 + Math.sin(3 * a + 1) * Math.sin(9 * y / 0.168 + 2 * a)) / 1.35 + 0.02 * (2 * random() - 1));
const meanRadius = (profile: Float32Array) => profile.reduce((sum, r) => sum + r, 0) / profile.length;

test("a clean can becomes a round shape of its own size", () => {
  const sample = sampleOf(revolved(0.168, () => noisy(0.033)));
  const fit = chooseShape(sample, 0.1);
  assert.equal(fit?.kind, "revolve");
  if (fit?.kind !== "revolve") return;
  assert.ok(Math.abs(meanRadius(fit.profile) - 0.033) < 0.001, `radius ${meanRadius(fit.profile)}`);
  assert.ok(Math.hypot(fit.center[0] - 0.02, fit.center[1] + 0.01) < 0.001, `axis ${fit.center}`);
  // straight up and down, not following the noise
  assert.equal(Math.max(...fit.profile), Math.min(...fit.profile));
});

test("a bottle keeps its shoulders and neck", () => {
  // 4 cm body, narrowing over the shoulder to a 1.2 cm neck
  const profile = (y: number) => (y < 0.15 ? 0.04 : y < 0.2 ? 0.04 - 0.028 * (y - 0.15) / 0.05 : 0.012);
  const fit = chooseShape(sampleOf(revolved(0.26, (y) => noisy(profile(y)), 12000)), 0.1);
  assert.equal(fit?.kind, "revolve");
  if (fit?.kind !== "revolve") return;
  const at = (y: number) => fit.profile[Math.floor(y / 0.26 * fit.profile.length)];
  assert.ok(Math.abs(at(0.05) - 0.04) < 0.002 && Math.abs(at(0.24) - 0.012) < 0.002, `body ${at(0.05)}, neck ${at(0.24)}`);
});

test("a shiny can's dents are filled out to its real radius, but only when it is shiny", () => {
  const sample = sampleOf(revolved(0.168, dented(0.033)));
  // as scanned, the median radius falls short of the can
  assert.ok(meanRadius(fitRevolve(sample)!.profile) < 0.031);
  const fit = chooseShape(sample, SHINY + 0.1);
  assert.equal(fit?.kind, "revolve");
  if (fit?.kind !== "revolve") return;
  assert.ok(Math.abs(meanRadius(fit.profile) - 0.033) < 0.0015, `radius ${meanRadius(fit.profile)}`);
  // dents fall unevenly over the height; the can stays straight anyway
  assert.equal(Math.max(...fit.profile), Math.min(...fit.profile));
  // the same lumps on a matte surface are the object's real shape: keep them
  assert.equal(chooseShape(sample, SHINY - 0.1), null);
});

test("a handle keeps a mug as scanned, shiny or not", () => {
  // a 4 cm mug with a handle reaching 2.5 cm out on one side over most of its height
  const mug = revolved(0.1, (y, a) => {
    const handle = Math.abs(Math.atan2(Math.sin(a), Math.cos(a))) < 0.35 && y > 0.015 && y < 0.085;
    return noisy(0.04) + (handle ? 0.025 * random() : 0);
  });
  assert.equal(chooseShape(sampleOf(mug), 0.1), null);
  assert.equal(chooseShape(sampleOf(mug), SHINY + 0.1), null);
});

test("a box becomes a box, turned the way it stands", () => {
  const sample = sampleOf(boxed([0.18, 0.12, 0.1], 0.5));
  const fit = chooseShape(sample, 0.1);
  assert.equal(fit?.kind, "box");
  if (fit?.kind !== "box") return;
  const [a, b] = [...fit.half].sort((x, y) => y - x);
  assert.ok(Math.abs(a - 0.09) < 0.003 && Math.abs(b - 0.05) < 0.003, `half ${fit.half}`);
  // the box's axes are 0.5 rad off x, z (up to a quarter turn)
  const off = ((fit.angle - 0.5) % (Math.PI / 2) + Math.PI / 2) % (Math.PI / 2);
  assert.ok(Math.min(off, Math.PI / 2 - off) < 0.03, `angle ${fit.angle}`);
  assert.ok(fitBox(sample)!.inliers > 0.95);
});

test("a box with a ball on top is not a simple shape", () => {
  const ball: Point[] = [];
  for (let i = 0; i < 2500; i++) {
    const z = random() * 2 - 1, a = random() * 2 * Math.PI, r = Math.sqrt(1 - z * z);
    const n: [number, number, number] = [r * Math.cos(a), z, r * Math.sin(a)];
    ball.push({ p: [0.04 * n[0], 0.12 + 0.04 + 0.04 * n[1], 0.04 * n[2]], n });
  }
  const sample = sampleOf([...boxed([0.18, 0.12, 0.14], 0.5), ...ball]);
  assert.equal(chooseShape(sample, 0.1), null);
  assert.equal(chooseShape(sample, SHINY + 0.1), null);
  assert.equal(mightBeShape(sampleOf(revolved(0.168, () => noisy(0.033)))), true);
});

/** Volume enclosed by a triangle mesh: positive when its faces wind outward. */
function volume(mesh: { positions: Float32Array; indices: Uint32Array }) {
  const p = mesh.positions;
  let sum = 0;
  for (let t = 0; t < mesh.indices.length; t += 3) {
    const [a, b, c] = [0, 1, 2].map((k) => mesh.indices[t + k] * 3);
    sum += p[a] * (p[b + 1] * p[c + 2] - p[b + 2] * p[c + 1]) - p[a + 1] * (p[b] * p[c + 2] - p[b + 2] * p[c]) + p[a + 2] * (p[b] * p[c + 1] - p[b + 1] * p[c]);
  }
  return sum / 6;
}

test("shape meshes are closed and face outward", () => {
  const can = chooseShape(sampleOf(revolved(0.168, () => noisy(0.033, 0.002))), 0.1)!;
  const round = shapeMesh(can);
  const expected = Math.PI * 0.033 ** 2 * can.top;
  assert.ok(Math.abs(volume(round) / expected - 1) < 0.02, `can volume ${volume(round)} vs ${expected}`);
  const box = chooseShape(sampleOf(boxed([0.18, 0.12, 0.1], 0.5)), 0.1)!;
  assert.equal(box.kind, "box");
  if (box.kind !== "box") return;
  const expectedBox = 4 * box.half[0] * box.half[1] * (box.top - box.bottom);
  assert.ok(Math.abs(volume(shapeMesh(box)) / expectedBox - 1) < 1e-3, `box volume ${volume(shapeMesh(box))} vs ${expectedBox}`);
});


test("assisted cylinder honors measured dimensions and rejects invalid input", () => {
  const fit = dimensionedCylinder(0.168, 0.066);
  const mesh = shapeMesh(fit);
  for (const [axis, expected] of [[0, 0.066], [1, 0.168], [2, 0.066]]) {
    const values = Array.from(mesh.positions).filter((_, i) => i % 3 === axis);
    assert.ok(Math.abs(Math.max(...values) - Math.min(...values) - expected) < 1e-7);
  }
  for (const invalid of [0, -1, NaN, Infinity]) {
    assert.throws(() => dimensionedCylinder(invalid, 0.066), RangeError);
    assert.throws(() => dimensionedCylinder(0.168, invalid), RangeError);
  }
});
