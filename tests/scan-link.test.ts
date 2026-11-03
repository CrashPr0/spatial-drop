import assert from 'node:assert/strict';
import test from 'node:test';
import { linkedViews } from '../pages-src/scan-link.js';

test('geometric checking accepts consistent views and rejects mismatched descriptors', () => {
  const n = 60, focal = 600;
  let seed = 71;
  const random = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 2 ** 32);
  const desc = Uint8Array.from({ length: n * 128 }, () => Math.floor(random() * 256));
  const a = { n, desc, x: new Float32Array(n), y: new Float32Array(n), w: 960, h: 720, focal };
  const b = { ...a, desc: desc.slice(), x: new Float32Array(n), y: new Float32Array(n) };
  for (let i = 0; i < n; i++) {
    const x = random() - 0.5, y = random() - 0.5, z = 2 + random();
    a.x[i] = 480 + focal * x / z; a.y[i] = 360 + focal * y / z;
    b.x[i] = 480 + focal * (x - 0.2) / z; b.y[i] = a.y[i];
  }
  assert.equal(linkedViews(a, b), true);
  assert.equal(linkedViews(a, { ...b, desc: new Uint8Array(n * 128) }), false);
  assert.equal(linkedViews({ ...a, n: 10 }, { ...b, n: 10 }), false);
});
