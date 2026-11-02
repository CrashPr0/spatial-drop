import assert from "node:assert/strict";
import test from "node:test";
import { cropToSubject, subjectBox, type CropFrame } from "../pages-src/scan-crop";

// a 0.4 m wide, 0.6 m tall upright cylinder at the origin, seen from 2 m away
// by a camera looking down +z (OpenCV axes: +y is down in the image)
const region = { center: [0, 0, 0] as [number, number, number], rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], radius: 0.2, bottom: -0.3, top: 0.3 };
const camera = { imgIdx: 0, R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: [0, 0, 2], f: 100 };
const intrinsics = { fx: 100, fy: 100, cx: 100, cy: 100 };

test("the subject box covers the cylinder's silhouette", () => {
  const box = subjectBox(camera, region, 200, 200, intrinsics, 1)!;
  // sides: tangent rays at asin(0.2 / 2) → 10.05 px; ends: the near rim at 1.8 m → 16.7 px
  assert.ok(box[0] <= 90 && box[0] >= 87, `left ${box[0]}`);
  assert.ok(box[2] >= 110 && box[2] <= 113, `right ${box[2]}`);
  assert.ok(box[1] <= 84 && box[1] >= 81, `top ${box[1]}`);
  assert.ok(box[3] >= 116 && box[3] <= 119, `bottom ${box[3]}`);
});

test("a turned camera finds the box where its pose says", () => {
  // camera at (2, 0, 0) looking back along −x; the cylinder moved 0.5 m along +z (image right)
  const R = [0, 0, 1, 0, 1, 0, -1, 0, 0];
  const C = [2, 0, 0];
  const t = [0, 1, 2].map((r) => -(R[r * 3] * C[0] + R[r * 3 + 1] * C[1] + R[r * 3 + 2] * C[2]));
  const box = subjectBox({ R, t, f: 100 }, { ...region, center: [0, 0, 0.5] }, 200, 200, intrinsics, 1)!;
  const middle = (box[0] + box[2]) / 2;
  assert.ok(Math.abs(middle - 125) < 2, `center column ${middle}`);
  assert.equal(subjectBox({ R, t, f: 100 }, { ...region, center: [0, 0, 5] }, 200, 200, intrinsics, 1), null);
});

test("cropping keeps every pixel where the camera model puts it", () => {
  // a 200 x 200 training frame (feature frame 400 x 400: s = 0.5) whose pixels encode their own position
  const rgb = new Float32Array(200 * 200 * 3);
  for (let y = 0; y < 200; y++) for (let x = 0; x < 200; x++) rgb.set([x / 255, y / 255, 0], (y * 200 + x) * 3);
  const frame: CropFrame = { fw: 400, fh: 400, tw: 200, th: 200, rgb };
  const cam = { ...camera, f: 200 };
  const kept = cropToSubject([frame], [cam], region, { margin: 0.02 });
  assert.ok(kept < 0.2, `kept ${kept}`);
  // the trainer's scale is unchanged, and a world point projects onto the same content
  assert.equal(frame.tw / frame.fw, 0.5);
  const project = (X: number[]) => [100 * X[0] / (X[2] + 2) + cam.cx! * 0.5, 100 * X[1] / (X[2] + 2) + cam.cy! * 0.5];
  for (const X of [[0, 0, 0], [0.15, -0.2, -0.1], [-0.18, 0.25, 0.1]]) {
    const [u, v] = project(X);
    const pixel = (Math.floor(v) * frame.tw + Math.floor(u)) * 3;
    const original = [100 * X[0] / (X[2] + 2) + 100, 100 * X[1] / (X[2] + 2) + 100];
    assert.equal(Math.round(frame.rgb![pixel] * 255), Math.floor(original[0]));
    assert.equal(Math.round(frame.rgb![pixel + 1] * 255), Math.floor(original[1]));
  }
});

test("frames mostly covered by the subject stay whole", () => {
  const frame: CropFrame = { fw: 200, fh: 200, tw: 200, th: 200, rgb: new Float32Array(200 * 200 * 3) };
  assert.equal(cropToSubject([frame], [{ ...camera, t: [0, 0, 0.25] }], region), 1);
  assert.equal(frame.tw, 200);
});
