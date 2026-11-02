import assert from "node:assert/strict";
import test from "node:test";
import { adviseCapture, matchFrames, shrink } from "../pages-src/scan-capture";

// a smooth random texture, wider than the frame so it can be panned
function scene(width: number, height: number, seed = 7) {
  let state = seed;
  const random = () => ((state = (state * 1664525 + 1013904223) >>> 0) / 2 ** 32);
  const coarse = Array.from({ length: (width / 4 + 2) * (height / 4 + 2) }, () => random() * 255);
  const cw = width / 4 + 2;
  const out = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const gx = x / 4, gy = y / 4, ix = Math.floor(gx), iy = Math.floor(gy), fx = gx - ix, fy = gy - iy;
      const at = (u: number, v: number) => coarse[(iy + v) * cw + ix + u];
      out[y * width + x] = (at(0, 0) * (1 - fx) + at(1, 0) * fx) * (1 - fy) + (at(0, 1) * (1 - fx) + at(1, 1) * fx) * fy;
    }
  }
  return out;
}

function view(world: Float32Array, worldWidth: number, left: number, top: number, width: number, height: number) {
  const out = new Float32Array(width * height);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) out[y * width + x] = world[(top + y) * worldWidth + left + x];
  return out;
}

const world = scene(120, 60);
const frame = (left: number, top = 10) => view(world, 120, left, top, 40, 30);

test("matching finds how far the view slid", () => {
  const match = matchFrames(frame(30), frame(38, 13), 40, 30);
  // the old content moved left and up in the new frame
  assert.equal(Math.round(match.dx * 40), -8);
  assert.equal(Math.round(match.dy * 30), -3);
  assert.ok(Math.abs(match.overlap - (32 * 27) / (40 * 30)) < 1e-9);
  assert.ok(match.score > 0.99, `score ${match.score}`);
});

test("advice: keep moving, then hold for a photo, then come back", () => {
  assert.equal(adviseCapture(null), "first");
  assert.equal(adviseCapture(matchFrames(frame(30), frame(32), 40, 30)), "move");
  assert.equal(adviseCapture(matchFrames(frame(30), frame(42), 40, 30)), "hold");
  // a different part of the world entirely
  assert.equal(adviseCapture(matchFrames(frame(0, 0), view(scene(40, 30, 99), 40, 0, 0, 40, 30), 40, 30)), "lost");
});

test("shrink averages blocks", () => {
  const small = shrink(new Float32Array([0, 2, 4, 6, 0, 2, 4, 6]), 4, 2, 2);
  assert.deepEqual([...small.gray], [1, 5]);
  assert.equal(small.width, 2);
  assert.equal(small.height, 1);
});
