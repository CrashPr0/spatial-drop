// geometry.js — minimal multi-view geometry toolbox (pure JS, no dependencies).
// Conventions: OpenCV-style camera (x right, y down, z forward).
// World-to-camera: pc = R * pw + t.  Normalized image coords: (x/z, y/z).

// ---------------------------------------------------------------------------
// Small dense linear algebra
// ---------------------------------------------------------------------------

/** Cyclic Jacobi eigen decomposition of a symmetric n*n matrix (row-major).
 *  Returns { vals, vecs } with eigenvalues sorted descending and eigenvectors
 *  as COLUMNS of vecs. */
export function jacobiEigen(Ain, n) {
  const A = Float64Array.from(Ain);
  const V = new Float64Array(n * n);
  for (let i = 0; i < n; i++) V[i * n + i] = 1;
  for (let sweep = 0; sweep < 80; sweep++) {
    let off = 0;
    for (let p = 0; p < n - 1; p++)
      for (let q = p + 1; q < n; q++) off += A[p * n + q] * A[p * n + q];
    if (off < 1e-26) break;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = A[p * n + q];
        if (Math.abs(apq) < 1e-30) continue;
        const app = A[p * n + p], aqq = A[q * n + q];
        const tau = (aqq - app) / (2 * apq);
        const t = (tau >= 0 ? 1 : -1) / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
        const c = 1 / Math.sqrt(1 + t * t), s = t * c;
        // A <- A * J
        for (let k = 0; k < n; k++) {
          const akp = A[k * n + p], akq = A[k * n + q];
          A[k * n + p] = c * akp - s * akq;
          A[k * n + q] = s * akp + c * akq;
        }
        // A <- J^T * A
        for (let k = 0; k < n; k++) {
          const apk = A[p * n + k], aqk = A[q * n + k];
          A[p * n + k] = c * apk - s * aqk;
          A[q * n + k] = s * apk + c * aqk;
        }
        // V <- V * J
        for (let k = 0; k < n; k++) {
          const vkp = V[k * n + p], vkq = V[k * n + q];
          V[k * n + p] = c * vkp - s * vkq;
          V[k * n + q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const idx = Array.from({ length: n }, (_, i) => i)
    .sort((a, b) => A[b * n + b] - A[a * n + a]);
  const vals = new Float64Array(n), vecs = new Float64Array(n * n);
  for (let j = 0; j < n; j++) {
    vals[j] = A[idx[j] * n + idx[j]];
    for (let i = 0; i < n; i++) vecs[i * n + j] = V[i * n + idx[j]];
  }
  return { vals, vecs };
}

/** Column j of an n*n matrix (row-major) as an array. */
export function matCol(M, n, j) {
  const c = new Float64Array(n);
  for (let i = 0; i < n; i++) c[i] = M[i * n + j];
  return c;
}

/** Solve A x = b (n*n, row-major) via Gaussian elimination with partial pivoting.
 *  Returns Float64Array x or null if singular. */
export function solveLinear(Ain, bin, n) {
  const A = Float64Array.from(Ain);
  const b = Float64Array.from(bin);
  for (let col = 0; col < n; col++) {
    let piv = col, best = Math.abs(A[col * n + col]);
    for (let r = col + 1; r < n; r++) {
      const v = Math.abs(A[r * n + col]);
      if (v > best) { best = v; piv = r; }
    }
    if (best < 1e-14) return null;
    if (piv !== col) {
      for (let k = col; k < n; k++) {
        const tmp = A[col * n + k]; A[col * n + k] = A[piv * n + k]; A[piv * n + k] = tmp;
      }
      const tb = b[col]; b[col] = b[piv]; b[piv] = tb;
    }
    const d = A[col * n + col];
    for (let r = col + 1; r < n; r++) {
      const f = A[r * n + col] / d;
      if (f === 0) continue;
      for (let k = col; k < n; k++) A[r * n + k] -= f * A[col * n + k];
      b[r] -= f * b[col];
    }
  }
  const x = new Float64Array(n);
  for (let r = n - 1; r >= 0; r--) {
    let s = b[r];
    for (let k = r + 1; k < n; k++) s -= A[r * n + k] * x[k];
    x[r] = s / A[r * n + r];
  }
  return x;
}

// ---------------------------------------------------------------------------
// 3x3 helpers (Float64Array(9), row-major) and rotations
// ---------------------------------------------------------------------------

export const I3 = () => Float64Array.from([1, 0, 0, 0, 1, 0, 0, 0, 1]);

export function m3mul(A, B) {
  const C = new Float64Array(9);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      C[i * 3 + j] = A[i * 3] * B[j] + A[i * 3 + 1] * B[3 + j] + A[i * 3 + 2] * B[6 + j];
  return C;
}

export function m3mulv(A, v) {
  return [
    A[0] * v[0] + A[1] * v[1] + A[2] * v[2],
    A[3] * v[0] + A[4] * v[1] + A[5] * v[2],
    A[6] * v[0] + A[7] * v[1] + A[8] * v[2],
  ];
}

export function m3t(A) {
  return Float64Array.from([A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]]);
}

export function m3det(A) {
  return A[0] * (A[4] * A[8] - A[5] * A[7])
       - A[1] * (A[3] * A[8] - A[5] * A[6])
       + A[2] * (A[3] * A[7] - A[4] * A[6]);
}

export const vsub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const vadd = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const vscale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
export const vdot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const vcross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
export const vnorm = (a) => Math.sqrt(vdot(a, a));
export function vnormalize(a) {
  const n = vnorm(a);
  return n > 1e-20 ? vscale(a, 1 / n) : [0, 0, 0];
}

/** Axis-angle (3-vector) -> rotation matrix. */
export function rodrigues(w) {
  const th = Math.sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
  if (th < 1e-12) return I3();
  const k = [w[0] / th, w[1] / th, w[2] / th];
  const c = Math.cos(th), s = Math.sin(th), v = 1 - c;
  return Float64Array.from([
    c + k[0] * k[0] * v,        k[0] * k[1] * v - k[2] * s, k[0] * k[2] * v + k[1] * s,
    k[1] * k[0] * v + k[2] * s, c + k[1] * k[1] * v,        k[1] * k[2] * v - k[0] * s,
    k[2] * k[0] * v - k[1] * s, k[2] * k[1] * v + k[0] * s, c + k[2] * k[2] * v,
  ]);
}

/** Rotation matrix -> axis-angle (3-vector). */
export function so3Log(R) {
  const tr = R[0] + R[4] + R[8];
  const c = Math.min(1, Math.max(-1, (tr - 1) / 2));
  const th = Math.acos(c);
  if (th < 1e-9) return [0, 0, 0];
  const s = 2 * Math.sin(th);
  return [
    (R[7] - R[5]) / s * th,
    (R[2] - R[6]) / s * th,
    (R[3] - R[1]) / s * th,
  ];
}

/** Project world point through pose {R, t}. Returns [xn, yn, z]. */
export function projectPoint(R, t, X) {
  const x = R[0] * X[0] + R[1] * X[1] + R[2] * X[2] + t[0];
  const y = R[3] * X[0] + R[4] * X[1] + R[5] * X[2] + t[1];
  const z = R[6] * X[0] + R[7] * X[1] + R[8] * X[2] + t[2];
  return [x / z, y / z, z];
}

// ---------------------------------------------------------------------------
// Triangulation (DLT on normalized coordinates)
// ---------------------------------------------------------------------------

/** Triangulate one point from N observations.
 *  cams: [{R, t}], obs: [[xn, yn]]. Returns [x, y, z] or null. */
export function triangulateN(cams, obs) {
  const AtA = new Float64Array(16);
  const row = new Float64Array(4);
  const addRow = () => {
    for (let i = 0; i < 4; i++)
      for (let j = i; j < 4; j++) AtA[i * 4 + j] += row[i] * row[j];
  };
  for (let k = 0; k < cams.length; k++) {
    const { R, t } = cams[k];
    const [u, v] = obs[k];
    // u * P3 - P1
    row[0] = u * R[6] - R[0]; row[1] = u * R[7] - R[1];
    row[2] = u * R[8] - R[2]; row[3] = u * t[2] - t[0];
    addRow();
    // v * P3 - P2
    row[0] = v * R[6] - R[3]; row[1] = v * R[7] - R[4];
    row[2] = v * R[8] - R[5]; row[3] = v * t[2] - t[1];
    addRow();
  }
  for (let i = 0; i < 4; i++)
    for (let j = 0; j < i; j++) AtA[i * 4 + j] = AtA[j * 4 + i];
  const { vecs } = jacobiEigen(AtA, 4);
  const h = matCol(vecs, 4, 3); // smallest eigenvalue -> last column
  if (Math.abs(h[3]) < 1e-12) return null;
  return [h[0] / h[3], h[1] / h[3], h[2] / h[3]];
}

/** Triangulation ray angle (radians) between two camera centers and X. */
export function parallaxAngle(cam1, cam2, X) {
  const c1 = vscale(m3mulv(m3t(cam1.R), cam1.t), -1);
  const c2 = vscale(m3mulv(m3t(cam2.R), cam2.t), -1);
  const r1 = vnormalize(vsub(X, c1));
  const r2 = vnormalize(vsub(X, c2));
  return Math.acos(Math.min(1, Math.max(-1, vdot(r1, r2))));
}

// ---------------------------------------------------------------------------
// Essential matrix: 8-point + RANSAC + decomposition
// ---------------------------------------------------------------------------

/** Least-squares essential matrix from normalized correspondences at the given
 *  sample indices. Enforces singular values (1, 1, 0). Returns {E, U, V} or null.
 *  Constraint form: x2^T E x1 = 0. */
export function estimateE(x1s, x2s, indices) {
  const AtA = new Float64Array(81);
  const a = new Float64Array(9);
  for (const idx of indices) {
    const [x1, y1] = x1s[idx], [x2, y2] = x2s[idx];
    a[0] = x2 * x1; a[1] = x2 * y1; a[2] = x2;
    a[3] = y2 * x1; a[4] = y2 * y1; a[5] = y2;
    a[6] = x1;      a[7] = y1;      a[8] = 1;
    for (let i = 0; i < 9; i++)
      for (let j = i; j < 9; j++) AtA[i * 9 + j] += a[i] * a[j];
  }
  for (let i = 0; i < 9; i++)
    for (let j = 0; j < i; j++) AtA[i * 9 + j] = AtA[j * 9 + i];
  const { vecs } = jacobiEigen(AtA, 9);
  const e = matCol(vecs, 9, 8);
  return enforceEssential(Float64Array.from(e));
}

/** Project a 3x3 matrix onto the essential manifold (singular values 1,1,0).
 *  Also returns consistent U, V columns for pose decomposition. */
export function enforceEssential(E) {
  const EtE = m3mul(m3t(E), E);
  const { vals, vecs } = jacobiEigen(EtE, 3);
  if (vals[1] <= 1e-18) return null;
  let v1 = matCol(vecs, 3, 0), v2 = matCol(vecs, 3, 1);
  let u1 = vnormalize(m3mulv(E, [v1[0], v1[1], v1[2]]));
  let u2raw = m3mulv(E, [v2[0], v2[1], v2[2]]);
  // Orthogonalize u2 against u1 for numerical safety.
  let u2 = vnormalize(vsub(u2raw, vscale(u1, vdot(u1, u2raw))));
  if (vnorm(u1) < 0.5 || vnorm(u2) < 0.5) return null;
  const u3 = vcross(u1, u2);
  const v3 = vcross([v1[0], v1[1], v1[2]], [v2[0], v2[1], v2[2]]);
  // E' = u1 v1^T + u2 v2^T
  const Ee = new Float64Array(9);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      Ee[i * 3 + j] = u1[i] * v1[j] + u2[i] * v2[j];
  const U = Float64Array.from([u1[0], u2[0], u3[0], u1[1], u2[1], u3[1], u1[2], u2[2], u3[2]]);
  const V = Float64Array.from([v1[0], v2[0], v3[0], v1[1], v2[1], v3[1], v1[2], v2[2], v3[2]]);
  return { E: Ee, U, V };
}

/** First-order Sampson error for x2^T E x1 = 0 (squared, in normalized units). */
export function sampsonError2(E, x1, x2) {
  const l2x = E[0] * x1[0] + E[1] * x1[1] + E[2]; // E x1
  const l2y = E[3] * x1[0] + E[4] * x1[1] + E[5];
  const l2z = E[6] * x1[0] + E[7] * x1[1] + E[8];
  const l1x = E[0] * x2[0] + E[3] * x2[1] + E[6]; // E^T x2
  const l1y = E[1] * x2[0] + E[4] * x2[1] + E[7];
  const err = x2[0] * l2x + x2[1] * l2y + l2z;
  const den = l2x * l2x + l2y * l2y + l1x * l1x + l1y * l1y;
  return den > 1e-20 ? (err * err) / den : Infinity;
}

/** RANSAC essential matrix. x1s/x2s: arrays of [xn, yn]. thresh: Sampson
 *  (normalized units, NOT squared). Returns {E, U, V, inliers} or null. */
export function ransacE(x1s, x2s, thresh, rng, maxIters = 1200, stats = null) {
  const n = x1s.length;
  if (n < 8) return null;
  let estFails = 0;
  const th2 = thresh * thresh;
  let best = null, bestCount = 0;
  let iters = maxIters;
  const sample = new Array(8);
  for (let it = 0; it < iters; it++) {
    // 8 distinct random indices
    for (let k = 0; k < 8; k++) {
      let cand;
      do {
        cand = (rng() * n) | 0;
      } while (sample.indexOf(cand) >= 0 && sample.indexOf(cand) < k);
      sample[k] = cand;
    }
    const est = estimateE(x1s, x2s, sample);
    if (!est) { estFails++; continue; }
    let count = 0;
    for (let i = 0; i < n; i++)
      if (sampsonError2(est.E, x1s[i], x2s[i]) < th2) count++;
    if (count > bestCount) {
      bestCount = count;
      best = est;
      // Adaptive termination — with the underflow guard. For a big pair (n > ~800)
      // whose FIRST sample is contaminated (most are: eight random points from a
      // 50 %-inlier set are all inliers 0.4 % of the time), count/n is under 1 %,
      // w^8 is below double epsilon, p rounds to exactly 1, log(p) is 0 and `need`
      // is -Infinity — the loop then ended after ONE iteration and the pair came
      // back with 3 inliers of 1,335. Every strong adjacent pair of Tom's orbit
      // died this way (2026-09-17, ?pairdebug=1: bestCount 1-7, iters 1-2), and
      // more iterations could not help because iters was pinned to it + 1.
      const w = count / n;
      const p = Math.max(1e-9, 1 - Math.pow(w, 8));
      const need = p >= 1 ? maxIters : Math.ceil(Math.log(1e-3) / Math.log(p));
      iters = Math.min(maxIters, Math.max(it + 1, need));
    }
  }
  if (stats) { stats.bestCount = bestCount; stats.estFails = estFails; stats.iters = iters; }
  if (!best || bestCount < 12) return null;
  // Refit on inliers, then recollect inliers.
  let inliers = [];
  for (let i = 0; i < n; i++)
    if (sampsonError2(best.E, x1s[i], x2s[i]) < th2) inliers.push(i);
  const refit = estimateE(x1s, x2s, inliers);
  if (refit) {
    let count = 0;
    const inl2 = [];
    for (let i = 0; i < n; i++)
      if (sampsonError2(refit.E, x1s[i], x2s[i]) < th2) { inl2.push(i); count++; }
    if (count >= bestCount * 0.8) { best = refit; inliers = inl2; }
  }
  return { ...best, inliers };
}

/** Decompose an enforced essential matrix (via its U, V) into 4 candidate
 *  relative poses {R, t} (cam1 = identity, cam2 = {R, t}, |t| = 1). */
export function decomposeE(U, V) {
  const W = Float64Array.from([0, -1, 0, 1, 0, 0, 0, 0, 1]);
  const Wt = m3t(W);
  const Vt = m3t(V);
  let Ra = m3mul(m3mul(U, W), Vt);
  let Rb = m3mul(m3mul(U, Wt), Vt);
  if (m3det(Ra) < 0) Ra = Ra.map((v) => -v);
  if (m3det(Rb) < 0) Rb = Rb.map((v) => -v);
  const u3 = [U[2], U[5], U[8]];
  const nu3 = vscale(u3, -1);
  return [
    { R: Ra, t: u3 }, { R: Ra, t: nu3 },
    { R: Rb, t: u3 }, { R: Rb, t: nu3 },
  ];
}

/** Pick the pose candidate with most points in front of both cameras.
 *  Returns {R, t, goodFrac, medianAngle, count} or null. */
export function selectPose(candidates, x1s, x2s, inliers) {
  const cam1 = { R: I3(), t: [0, 0, 0] };
  const subset = inliers.length > 300
    ? inliers.filter((_, i) => i % Math.ceil(inliers.length / 300) === 0)
    : inliers;
  let best = null;
  for (const cand of candidates) {
    const cam2 = { R: cand.R, t: cand.t };
    let good = 0;
    const angles = [];
    for (const idx of subset) {
      const X = triangulateN([cam1, cam2], [x1s[idx], x2s[idx]]);
      if (!X) continue;
      const z1 = X[2];
      const z2 = cand.R[6] * X[0] + cand.R[7] * X[1] + cand.R[8] * X[2] + cand.t[2];
      if (z1 > 0 && z2 > 0) {
        good++;
        angles.push(parallaxAngle(cam1, cam2, X));
      }
    }
    if (!best || good > best.count) {
      angles.sort((a, b) => a - b);
      best = {
        R: cand.R, t: cand.t, count: good,
        goodFrac: good / Math.max(1, subset.length),
        medianAngle: angles.length ? angles[(angles.length / 2) | 0] : 0,
      };
    }
  }
  return best;
}

// ---------------------------------------------------------------------------
// PnP: DLT + RANSAC + Levenberg-Marquardt pose refinement
// ---------------------------------------------------------------------------

/** DLT pose from >= 6 (world point, normalized image point) correspondences.
 *  Returns {R, t} or null. */
export function dltPnP(objPts, imgPts, indices) {
  const m = indices.length;
  if (m < 6) return null;
  // Normalize 3D points for conditioning.
  let cx = 0, cy = 0, cz = 0;
  for (const i of indices) { cx += objPts[i][0]; cy += objPts[i][1]; cz += objPts[i][2]; }
  cx /= m; cy /= m; cz /= m;
  let scale = 0;
  for (const i of indices) {
    const d = [objPts[i][0] - cx, objPts[i][1] - cy, objPts[i][2] - cz];
    scale += vnorm(d);
  }
  scale = scale / m || 1;
  const s = Math.sqrt(3) / scale;

  const AtA = new Float64Array(144);
  const row = new Float64Array(12);
  const acc = () => {
    for (let i = 0; i < 12; i++)
      for (let j = i; j < 12; j++) AtA[i * 12 + j] += row[i] * row[j];
  };
  for (const i of indices) {
    const X = (objPts[i][0] - cx) * s, Y = (objPts[i][1] - cy) * s, Z = (objPts[i][2] - cz) * s;
    const [u, v] = imgPts[i];
    row.fill(0);
    row[0] = X; row[1] = Y; row[2] = Z; row[3] = 1;
    row[8] = -u * X; row[9] = -u * Y; row[10] = -u * Z; row[11] = -u;
    acc();
    row.fill(0);
    row[4] = X; row[5] = Y; row[6] = Z; row[7] = 1;
    row[8] = -v * X; row[9] = -v * Y; row[10] = -v * Z; row[11] = -v;
    acc();
  }
  for (let i = 0; i < 12; i++)
    for (let j = 0; j < i; j++) AtA[i * 12 + j] = AtA[j * 12 + i];
  const { vecs } = jacobiEigen(AtA, 12);
  const p = matCol(vecs, 12, 11);
  let M = Float64Array.from([p[0], p[1], p[2], p[4], p[5], p[6], p[8], p[9], p[10]]);
  let tv = [p[3], p[7], p[11]];
  let d = m3det(M);
  if (Math.abs(d) < 1e-18) return null;
  if (d < 0) { M = M.map((v) => -v); tv = vscale(tv, -1); d = -d; }
  const sc = Math.cbrt(d);
  M = M.map((v) => v / sc);
  tv = vscale(tv, 1 / sc);
  // Polar decomposition: R = M (M^T M)^(-1/2)
  const MtM = m3mul(m3t(M), M);
  const { vals, vecs: Q } = jacobiEigen(MtM, 3);
  if (vals[2] <= 1e-14) return null;
  const invSqrt = new Float64Array(9);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++) {
      let sum = 0;
      for (let k = 0; k < 3; k++)
        sum += Q[i * 3 + k] * Q[j * 3 + k] / Math.sqrt(vals[k]);
      invSqrt[i * 3 + j] = sum;
    }
  let R = m3mul(M, invSqrt);
  if (m3det(R) < 0) return null;
  // Undo 3D normalization: pc = R * s * (pw - c) + t  =>  t_w = t - s * R * c... derive:
  // Estimated maps X' = s (X - c). pc = R X' + tv = s R X + (tv - s R c).
  // Renormalize scale so rotation stays orthonormal: pose defined up to none —
  // projective DLT already fixed overall scale via det; final translation:
  const Rc = m3mulv(R, [cx, cy, cz]);
  const t = [tv[0] / s - Rc[0] + 0, tv[1] / s - Rc[1], tv[2] / s - Rc[2]];
  // Explanation: pc = s R X + tv - s R c  => divide by s (projection invariant):
  // pc' = R X + (tv - s R c)/s = R X + tv/s - R c.
  return { R, t };
}

/** Reprojection error (normalized units) of pose on one correspondence. */
export function reprojError(R, t, X, xn) {
  const z = R[6] * X[0] + R[7] * X[1] + R[8] * X[2] + t[2];
  if (z <= 1e-6) return Infinity;
  const u = (R[0] * X[0] + R[1] * X[1] + R[2] * X[2] + t[0]) / z;
  const v = (R[3] * X[0] + R[4] * X[1] + R[5] * X[2] + t[1]) / z;
  return Math.hypot(u - xn[0], v - xn[1]);
}

/** Levenberg-Marquardt pose refinement with numeric Jacobians.
 *  objPts/imgPts: arrays; returns refined {R, t}. */
export function refinePose(R0, t0, objPts, imgPts, iters = 10) {
  let R = Float64Array.from(R0);
  let t = t0.slice();
  const n = objPts.length;
  if (n < 4) return { R, t };

  const residuals = (Rx, tx, out) => {
    let cost = 0;
    for (let i = 0; i < n; i++) {
      const X = objPts[i];
      const z = Rx[6] * X[0] + Rx[7] * X[1] + Rx[8] * X[2] + tx[2];
      let ru, rv;
      if (z <= 1e-6) { ru = 0.5; rv = 0.5; }
      else {
        ru = (Rx[0] * X[0] + Rx[1] * X[1] + Rx[2] * X[2] + tx[0]) / z - imgPts[i][0];
        rv = (Rx[3] * X[0] + Rx[4] * X[1] + Rx[5] * X[2] + tx[1]) / z - imgPts[i][1];
      }
      // mild robustness: clamp huge residuals (soft enough that LM still
      // gets gradient from a coarse initial pose)
      ru = Math.max(-0.6, Math.min(0.6, ru));
      rv = Math.max(-0.6, Math.min(0.6, rv));
      if (out) { out[2 * i] = ru; out[2 * i + 1] = rv; }
      cost += ru * ru + rv * rv;
    }
    return cost;
  };

  const r0 = new Float64Array(2 * n);
  const rp = new Float64Array(2 * n);
  const J = new Float64Array(2 * n * 6);
  let lambda = 1e-4;
  let cost = residuals(R, t, r0);

  for (let it = 0; it < iters; it++) {
    // numeric Jacobian around current pose (local perturbation)
    const eps = 1e-6;
    for (let k = 0; k < 6; k++) {
      let Rp = R, tp = t;
      if (k < 3) {
        const w = [0, 0, 0]; w[k] = eps;
        Rp = m3mul(rodrigues(w), R);
      } else {
        tp = t.slice(); tp[k - 3] += eps;
      }
      residuals(Rp, tp, rp);
      for (let i = 0; i < 2 * n; i++) J[i * 6 + k] = (rp[i] - r0[i]) / eps;
    }
    // normal equations
    const H = new Float64Array(36), g = new Float64Array(6);
    for (let i = 0; i < 2 * n; i++) {
      for (let a = 0; a < 6; a++) {
        g[a] += J[i * 6 + a] * r0[i];
        for (let b = a; b < 6; b++) H[a * 6 + b] += J[i * 6 + a] * J[i * 6 + b];
      }
    }
    for (let a = 0; a < 6; a++)
      for (let b = 0; b < a; b++) H[a * 6 + b] = H[b * 6 + a];

    let improved = false;
    for (let attempt = 0; attempt < 4; attempt++) {
      const Hd = Float64Array.from(H);
      for (let a = 0; a < 6; a++) Hd[a * 6 + a] += lambda * (1 + H[a * 6 + a]);
      const negG = g.map((v) => -v);
      const delta = solveLinear(Hd, negG, 6);
      if (!delta) { lambda *= 10; continue; }
      const Rn = m3mul(rodrigues([delta[0], delta[1], delta[2]]), R);
      const tn = [t[0] + delta[3], t[1] + delta[4], t[2] + delta[5]];
      const cn = residuals(Rn, tn, null);
      if (cn < cost) {
        R = Rn; t = tn; cost = cn;
        residuals(R, t, r0);
        lambda = Math.max(1e-8, lambda * 0.3);
        improved = true;
        break;
      }
      lambda *= 10;
    }
    if (!improved) break;
  }
  return { R, t };
}

/** RANSAC PnP. Returns {R, t, inliers} or null. thresh in normalized units. */
export function pnpRansac(objPts, imgPts, thresh, rng, maxIters = 500) {
  const n = objPts.length;
  if (n < 6) return null;
  let best = null, bestCount = 0;
  let iters = maxIters;
  const sample = new Array(6);
  for (let it = 0; it < iters; it++) {
    for (let k = 0; k < 6; k++) {
      let cand;
      do {
        cand = (rng() * n) | 0;
      } while (sample.indexOf(cand) >= 0 && sample.indexOf(cand) < k);
      sample[k] = cand;
    }
    const pose = dltPnP(objPts, imgPts, sample);
    if (!pose) continue;
    let count = 0;
    for (let i = 0; i < n; i++)
      if (reprojError(pose.R, pose.t, objPts[i], imgPts[i]) < thresh) count++;
    if (count > bestCount) {
      bestCount = count;
      best = pose;
      const w = count / n;
      const p = Math.max(1e-9, 1 - Math.pow(w, 6));
      const need = Math.ceil(Math.log(1e-3) / Math.log(p));
      iters = Math.min(maxIters, Math.max(it + 1, need));
    }
  }
  if (!best || bestCount < 6) return null;
  // refine on inliers
  const inObj = [], inImg = [];
  for (let i = 0; i < n; i++)
    if (reprojError(best.R, best.t, objPts[i], imgPts[i]) < thresh) {
      inObj.push(objPts[i]); inImg.push(imgPts[i]);
    }
  const refined = refinePose(best.R, best.t, inObj, inImg, 12);
  const inliers = [];
  for (let i = 0; i < n; i++)
    if (reprojError(refined.R, refined.t, objPts[i], imgPts[i]) < thresh) inliers.push(i);
  if (inliers.length < 6) return null;
  return { R: refined.R, t: refined.t, inliers };
}

/** All roots of a quartic via Durand-Kerner; returns the real ones. */
function quarticRealRoots(a4, a3, a2, a1, a0) {
  const scale = Math.max(Math.abs(a4), Math.abs(a3), Math.abs(a2), Math.abs(a1), Math.abs(a0));
  if (!(scale > 0)) return [];
  if (Math.abs(a4) < 1e-13 * scale) {
    // degenerate leading term: fall back to the cubic/quadratic inside
    if (Math.abs(a3) < 1e-13 * scale) {
      if (Math.abs(a2) < 1e-13 * scale) return Math.abs(a1) > 0 ? [-a0 / a1] : [];
      const D = a1 * a1 - 4 * a2 * a0;
      if (D < 0) return [];
      const s = Math.sqrt(D);
      return [(-a1 + s) / (2 * a2), (-a1 - s) / (2 * a2)];
    }
    // cubic: one guaranteed real root by bisection + deflate to quadratic
    const f = (x) => ((a3 * x + a2) * x + a1) * x + a0;
    let lo = -1, hi = 1;
    while (f(lo) * f(hi) > 0 && hi < 1e12) { lo *= 2; hi *= 2; }
    if (f(lo) * f(hi) > 0) return [];
    for (let i = 0; i < 200; i++) {
      const m = (lo + hi) / 2;
      if (f(lo) * f(m) <= 0) hi = m; else lo = m;
    }
    const r = (lo + hi) / 2;
    const qa = a3, qb = a2 + qa * r, qc = a1 + qb * r;
    const D = qb * qb - 4 * qa * qc;
    const out = [r];
    if (D >= 0) { const s = Math.sqrt(D); out.push((-qb + s) / (2 * qa), (-qb - s) / (2 * qa)); }
    return out;
  }
  const c = [a3 / a4, a2 / a4, a1 / a4, a0 / a4];
  // Durand-Kerner on z^4 + c0 z^3 + c1 z^2 + c2 z + c3
  const zr = [1, -0.4, 0.3, -1.1], zi = [0.9, 1.1, -1.2, -0.7];
  const ev = (x, y) => {   // p(x+iy)
    let pr = 1, pi = 0;
    for (let k = 0; k < 4; k++) {
      const nr = pr * x - pi * y + c[k];
      pi = pr * y + pi * x;
      pr = nr;
    }
    return [pr, pi];
  };
  for (let it = 0; it < 80; it++) {
    let moved = 0;
    for (let i = 0; i < 4; i++) {
      const [pr, pi] = ev(zr[i], zi[i]);
      let dr = 1, di = 0;
      for (let j = 0; j < 4; j++) {
        if (j === i) continue;
        const ax = zr[i] - zr[j], ay = zi[i] - zi[j];
        const nr = dr * ax - di * ay, ni = dr * ay + di * ax;
        dr = nr; di = ni;
      }
      const den = dr * dr + di * di || 1e-30;
      const qr = (pr * dr + pi * di) / den, qi = (pi * dr - pr * di) / den;
      zr[i] -= qr; zi[i] -= qi;
      moved = Math.max(moved, Math.hypot(qr, qi));
    }
    if (moved < 1e-14) break;
  }
  const out = [];
  for (let i = 0; i < 4; i++)
    if (Math.abs(zi[i]) < 1e-6 * (1 + Math.abs(zr[i]))) out.push(zr[i]);
  return out;
}

/** P3P for a central camera on unit BEARING vectors (any direction, not just
 *  a forward-facing pinhole) — Grunert's quartic + Horn absolute orientation.
 *  P: three world points; brg: three unit bearings in the camera/rig frame.
 *  Returns candidate poses [{R, C}] with R world->camera and C the centre
 *  (pw maps to R*(pw - C)); [] when degenerate. */
export function p3pBearings(P, brg) {
  const d12 = [P[0][0] - P[1][0], P[0][1] - P[1][1], P[0][2] - P[1][2]];
  const d13 = [P[0][0] - P[2][0], P[0][1] - P[2][1], P[0][2] - P[2][2]];
  const d23 = [P[1][0] - P[2][0], P[1][1] - P[2][1], P[1][2] - P[2][2]];
  const a = Math.hypot(...d23), b = Math.hypot(...d13), cc = Math.hypot(...d12);
  if (a < 1e-12 || b < 1e-12 || cc < 1e-12) return [];
  // collinear world points span no orientation
  const cx = d12[1] * d13[2] - d12[2] * d13[1];
  const cy = d12[2] * d13[0] - d12[0] * d13[2];
  const cz = d12[0] * d13[1] - d12[1] * d13[0];
  if (Math.hypot(cx, cy, cz) < 1e-8 * b * cc) return [];
  const dot = (u, v) => u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
  const cosA = dot(brg[1], brg[2]);   // angle subtended by P2,P3
  const cosB = dot(brg[0], brg[2]);   // by P1,P3
  const cosG = dot(brg[0], brg[1]);   // by P1,P2
  const aSq = a * a, bSq = b * b, cSq = cc * cc;
  const q1 = (aSq - cSq) / bSq, q2 = (aSq + cSq) / bSq;
  const A4 = (q1 - 1) ** 2 - 4 * (cSq / bSq) * cosA * cosA;
  const A3 = 4 * (q1 * (1 - q1) * cosB - (1 - q2) * cosA * cosG + 2 * (cSq / bSq) * cosA * cosA * cosB);
  const A2 = 2 * (q1 * q1 - 1 + 2 * q1 * q1 * cosB * cosB + 2 * ((bSq - cSq) / bSq) * cosA * cosA
    - 4 * q2 * cosA * cosB * cosG + 2 * ((bSq - aSq) / bSq) * cosG * cosG);
  const A1 = 4 * (-q1 * (1 + q1) * cosB + 2 * (aSq / bSq) * cosG * cosG * cosB - (1 - q2) * cosA * cosG);
  const A0 = (1 + q1) ** 2 - 4 * (aSq / bSq) * cosG * cosG;
  const out = [];
  for (const v of quarticRealRoots(A4, A3, A2, A1, A0)) {
    if (!(v > 0)) continue;
    const s1sq = bSq / (1 + v * v - 2 * v * cosB);
    if (!(s1sq > 0)) continue;
    const s1 = Math.sqrt(s1sq);
    const s3 = v * s1;
    // s2 from the c-side law of cosines; the quartic already tied it, so of
    // the two quadratic roots keep the one(s) consistent with the a-side
    const disc = s1sq * cosG * cosG - s1sq + cSq;
    if (disc < 0) continue;
    const sd = Math.sqrt(disc);
    for (const s2 of [s1 * cosG + sd, s1 * cosG - sd]) {
      if (!(s2 > 0)) continue;
      const aErr = Math.abs(s2 * s2 + s3 * s3 - 2 * s2 * s3 * cosA - aSq);
      if (aErr > 1e-3 * aSq) continue;
      // absolute orientation: camera-frame points Q_i = s_i * brg_i vs P_i
      const Q = [
        [s1 * brg[0][0], s1 * brg[0][1], s1 * brg[0][2]],
        [s2 * brg[1][0], s2 * brg[1][1], s2 * brg[1][2]],
        [s3 * brg[2][0], s3 * brg[2][1], s3 * brg[2][2]],
      ];
      const Pc = [(P[0][0] + P[1][0] + P[2][0]) / 3, (P[0][1] + P[1][1] + P[2][1]) / 3, (P[0][2] + P[1][2] + P[2][2]) / 3];
      const Qc = [(Q[0][0] + Q[1][0] + Q[2][0]) / 3, (Q[0][1] + Q[1][1] + Q[2][1]) / 3, (Q[0][2] + Q[1][2] + Q[2][2]) / 3];
      let Sxx = 0, Sxy = 0, Sxz = 0, Syx = 0, Syy = 0, Syz = 0, Szx = 0, Szy = 0, Szz = 0;
      for (let i = 0; i < 3; i++) {
        const px = P[i][0] - Pc[0], py = P[i][1] - Pc[1], pz = P[i][2] - Pc[2];
        const qx = Q[i][0] - Qc[0], qy = Q[i][1] - Qc[1], qz = Q[i][2] - Qc[2];
        Sxx += px * qx; Sxy += px * qy; Sxz += px * qz;
        Syx += py * qx; Syy += py * qy; Syz += py * qz;
        Szx += pz * qx; Szy += pz * qy; Szz += pz * qz;
      }
      const N = [
        Sxx + Syy + Szz, Syz - Szy, Szx - Sxz, Sxy - Syx,
        Syz - Szy, Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz,
        Szx - Sxz, Sxy + Syx, -Sxx + Syy - Szz, Syz + Szy,
        Sxy - Syx, Szx + Sxz, Syz + Szy, -Sxx - Syy + Szz,
      ];
      const { vecs } = jacobiEigen(N, 4);
      const w = vecs[0], x = vecs[4], y = vecs[8], z = vecs[12];
      const R = [
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
      ];
      const C = [
        Pc[0] - (R[0] * Qc[0] + R[3] * Qc[1] + R[6] * Qc[2]),
        Pc[1] - (R[1] * Qc[0] + R[4] * Qc[1] + R[7] * Qc[2]),
        Pc[2] - (R[2] * Qc[0] + R[5] * Qc[1] + R[8] * Qc[2]),
      ];
      out.push({ R, C });
    }
  }
  return out;
}

/** Simple seedable PRNG (mulberry32). */
export function makeRng(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
