import assert from "node:assert/strict";
import test from "node:test";
import { adviseOrbit, adviseCapture, matchFrames, pointTo, shrink, type FrameMatch } from "../pages-src/scan-capture";

// a smooth random texture, wider than the frame so it can be panned
function scene(width: number, height: number, seed = 7) {
  let state = seed;
  const random = () => ((state = (state * 1664525 + 1013904223) >>> 0) / 2 ** 32);
  const cw = width / 4 + 2;
  const coarse = Array.from({ length: cw * (height / 4 + 2) }, () => random() * 255);
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

const world = scene(240, 120);
const frame = (left: number, top = 20) => view(world, 240, left, top, 80, 60);
const match = (overlap: number, dx = 0): FrameMatch => ({ dx, dy: 0, overlap, score: 0.9 });

test("matching finds how far the view slid and how much is still shared", () => {
  const found = matchFrames(frame(60), frame(68, 23), 80, 60);
  // the old content moved left and up in the new frame
  assert.equal(Math.round(found.dx * 80), -8);
  assert.equal(Math.round(found.dy * 60), -3);
  // 16 px tiles: the left column and top row slid out of view, 8 of 15 remain
  assert.ok(Math.abs(found.overlap - 8 / 15) < 1e-9, `overlap ${found.overlap}`);
  assert.ok(found.score > 0.99, `score ${found.score}`);
});

test("an unrelated view shares nothing", () => {
  const found = matchFrames(frame(0, 0), view(scene(80, 60, 99), 80, 0, 0, 80, 60), 80, 60);
  assert.ok(found.overlap < 0.1, `overlap ${found.overlap}`);
});

test("advice: keep moving, then hold for a photo, then come back", () => {
  assert.equal(adviseCapture(null), "first");
  assert.equal(adviseCapture(match(0.8)), "move");
  assert.equal(adviseCapture(match(0.3)), "hold");
  assert.equal(adviseCapture(match(0.05)), "lost");
});

test("the arrow keeps the orbit going and points back when lost", () => {
  assert.equal(pointTo("move", match(0.8), 0, 80), "either");
  assert.equal(pointTo("move", match(0.8), 1, 80), "right");
  assert.equal(pointTo("move", match(0.8), -1, 80), "left");
  assert.equal(pointTo("hold", match(0.3), 1, 80), null);
  // the last photo's tiles slid right: it is back to the left
  assert.equal(pointTo("lost", match(0.05, 6 / 80), 1, 80), "left");
  // nothing found: back against the heading
  assert.equal(pointTo("lost", match(0), -1, 80), "right");
});

test("shrink averages blocks", () => {
  const small = shrink(new Float32Array([0, 2, 4, 6, 0, 2, 4, 6]), 4, 2, 2);
  assert.deepEqual([...small.gray], [1, 5]);
  assert.equal(small.width, 2);
  assert.equal(small.height, 1);
});


test("capture requests a photo before most textured tiles are lost", () => {
  assert.equal(adviseCapture(match(0.55)), "hold");
  assert.equal(adviseCapture(match(0.4)), "hold");
  assert.equal(adviseCapture(match(0.7)), "move");
  assert.equal(adviseCapture(match(0.05)), "lost");
});


test("orbit guidance requests small steps and blocks large gaps", () => {
  assert.equal(adviseOrbit(match(0.9), 12, 0), "hold");
  assert.equal(adviseOrbit(match(0.9), 2, 350), "hold");
  assert.equal(adviseOrbit(match(0.9), 67, 0), "lost");
  assert.equal(adviseOrbit(match(0.05), 12, 0), "lost");
  assert.equal(adviseOrbit(match(0.9), null, 0), "move");
  assert.equal(adviseOrbit(null, 40, 0), "first");
});

test("with a heading, photos wait for real travel unless the view is slipping away", () => {
  // close to the object the tiles fall under the keep mark within a few degrees
  assert.equal(adviseCapture(match(0.5)), "hold");
  assert.equal(adviseOrbit(match(0.5), 4, 0), "move");
  assert.equal(adviseOrbit(match(0.5), 9, 0), "hold");
  // moving sideways without turning: the heading says nothing, the image does
  assert.equal(adviseOrbit(match(0.2), 4, 0), "hold");
  assert.equal(adviseOrbit(match(0.05), 4, 0), "lost");
});
