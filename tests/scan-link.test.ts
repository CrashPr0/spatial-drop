import assert from 'node:assert/strict';
import test from 'node:test';
import { LinkGate, linkedViews } from '../pages-src/scan-link.js';

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

// Views that link when they share a group, like photos from one stretch of a walk.
const view = (name: string, group = 'a') => ({ name, group });
const sameGroup = (a: { group: string }, b: { group: string }) => a.group === b.group;

test('a linked candidate is kept and joins the recent views', () => {
  const gate = new LinkGate(sameGroup);
  assert.deepEqual(gate.check(view('0')), { accepted: true, promoted: [], forced: false });
  assert.equal(gate.check(view('1')).accepted, true);
});

test('after a jump the gate picks up the new stretch instead of refusing it forever', () => {
  // the real can scan: three jumps in 58 photos, and without this every photo after the first was lost
  const gate = new LinkGate(sameGroup);
  gate.check(view('0'));
  assert.deepEqual(gate.check(view('1', 'b')), { accepted: false, promoted: [], forced: false });
  // the next photo links to the rejected one: both are a real stretch, so both are kept
  assert.deepEqual(gate.check(view('2', 'b')), { accepted: true, promoted: ['1'], forced: false });
  assert.deepEqual(gate.check(view('3', 'b')), { accepted: true, promoted: [], forced: false });
});

test('photos that link to nothing cannot stall a scan', () => {
  const gate = new LinkGate(sameGroup);
  gate.check(view('0'));
  assert.equal(gate.check(view('1', 'b')).accepted, false);
  assert.equal(gate.check(view('2', 'c')).accepted, false);
  assert.deepEqual(gate.check(view('3', 'd')), { accepted: true, promoted: [], forced: true });
  // and the forced view is a new place to link from
  assert.equal(gate.check(view('4', 'd')).accepted, true);
});

test('a new step starts its own chain', () => {
  const gate = new LinkGate(sameGroup);
  gate.check(view('0'));
  gate.check(view('1', 'b'));
  assert.deepEqual(gate.check(view('2', 'c'), { newChain: true }), { accepted: true, promoted: [], forced: false });
  assert.equal(gate.check(view('3', 'c')).accepted, true);
  // the rejected photo from the old step is gone: it cannot come back later
  assert.deepEqual(gate.check(view('4', 'b')), { accepted: false, promoted: [], forced: false });
});

test('seeded views stay linkable as new photos come in', () => {
  // Add detail: a photo anywhere on the old scan should link, however many were added since
  const gate = new LinkGate(sameGroup);
  gate.seed(view('old', 'seed'));
  for (let i = 0; i < 5; i++) gate.check(view(`new${i}`, 'seed'));
  gate.check(view('elsewhere', 'b'));
  assert.equal(gate.check(view('back', 'seed')).accepted, true);
});
