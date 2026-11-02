// pairworker.js — essential-matrix RANSAC for a batch of image pairs in a Web
// Worker. The per-pair geometry check (up to 600 RANSAC iterations, Sampson
// error over every match) used to run on the main thread pair after pair:
// 60 s of the precise Truck solve on a desktop, and a frozen page on phones.
// Pairs are independent, so a pool of these does the same work in parallel;
// each pair carries its own RNG seed so results do not depend on the split.
import { ransacE, makeRng } from './geometry.js';

self.onmessage = (e) => {
  const { batch } = e.data;   // [{ idx, x1, x2 (Float32Array, 2 per match), thresh, seed, maxIters }]
  const out = [];
  const buffers = [];
  for (const p of batch) {
    const m = p.x1.length >> 1;
    const x1s = new Array(m), x2s = new Array(m);
    for (let k = 0; k < m; k++) {
      x1s[k] = [p.x1[2 * k], p.x1[2 * k + 1]];
      x2s[k] = [p.x2[2 * k], p.x2[2 * k + 1]];
    }
    const stats = p.debug ? {} : null;
    const res = ransacE(x1s, x2s, p.thresh, makeRng(p.seed), p.maxIters, stats);
    const inliers = res ? Int32Array.from(res.inliers) : null;
    out.push({ idx: p.idx, inliers, stats });
    if (inliers) buffers.push(inliers.buffer);
  }
  self.postMessage({ out }, buffers);
};
