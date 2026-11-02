import assert from "node:assert/strict";
import test from "node:test";
import { neighborSpacing, pointSplats } from "../pages-src/scan-points";

// a 5 x 5 x 5 lattice, 1 cm apart
const lattice: [number, number, number][] = [];
for (let x = 0; x < 5; x++) for (let y = 0; y < 5; y++) for (let z = 0; z < 5; z++) lattice.push([x * 0.01, y * 0.01, z * 0.01]);

test("neighbor spacing on a lattice is the lattice step", () => {
  const spacing = neighborSpacing(lattice, 3);
  for (const value of spacing) assert.ok(Math.abs(value - 0.01) < 1e-9, `spacing ${value}`);
});

test("a lone far point does not get a huge splat", () => {
  const points = [...lattice, [2, 2, 2] as [number, number, number]].map((X) => ({ X, rgb: [0.5, 0.5, 0.5] }));
  const cloud = pointSplats(points, { k: 3, size: 0.3 });
  const scale = (i: number) => Math.exp(cloud.data[i * 16 + 3]);
  assert.ok(Math.abs(scale(0) - 0.003) < 1e-6, `lattice splat ${scale(0)}`);
  assert.ok(scale(points.length - 1) <= scale(0) + 1e-9, `stray splat ${scale(points.length - 1)}`);
});

test("splats keep position, color and opacity in the trainer layout", () => {
  const cloud = pointSplats([{ X: [1, 2, 3], rgb: [1, 0.5, 0] }, ...lattice.slice(0, 4).map((X) => ({ X, rgb: [0, 0, 0] }))]);
  assert.equal(cloud.dc, "sh");
  assert.deepEqual([...cloud.data.subarray(0, 3)], [1, 2, 3]);
  const SH_C0 = 0.28209479177387814;
  const color = [10, 11, 12].map((k) => 0.5 + SH_C0 * cloud.data[k]);
  color.forEach((c, i) => assert.ok(Math.abs(c - [1, 0.5, 0][i]) < 1e-6));
  assert.ok(Math.abs(1 / (1 + Math.exp(-cloud.data[13])) - 0.9) < 1e-6);
  assert.equal(cloud.data[6], 1);
});
