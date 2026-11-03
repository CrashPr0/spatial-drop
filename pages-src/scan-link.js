import { matchSift } from './vendor/splat/sfm/sift.js';
import { ransacE, makeRng } from './vendor/splat/sfm/geometry.js';

// A conservative early connection check, not a replacement for the full solve.
export function linkedViews(a, b) {
  const pairs = matchSift(a.desc, a.n, b.desc, b.n);
  if (pairs.length < 48) return false;
  const normalized = (f, i) => [(f.x[i] - f.w / 2) / f.focal, (f.y[i] - f.h / 2) / f.focal];
  const x1 = [], x2 = [];
  for (let i = 0; i < pairs.length; i += 2) {
    x1.push(normalized(a, pairs[i])); x2.push(normalized(b, pairs[i + 1]));
  }
  const result = ransacE(x1, x2, 2 / Math.min(a.focal, b.focal), makeRng(71), 200);
  return !!result && result.inliers.length >= 20 && result.inliers.length >= x1.length * 0.4;
}
