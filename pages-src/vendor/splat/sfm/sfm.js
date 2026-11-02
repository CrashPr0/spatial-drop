// sfm.js — minimal incremental Structure-from-Motion.
//
// features -> pairwise matching -> tracks (from raw mutual matches)
// -> focal search: for each candidate focal length, two-view initialization
//    (E-RANSAC over track correspondences) + incremental registration
// -> best reconstruction wins.
//
// Raw (ratio+cross-checked) matches build the track graph directly; geometric
// filtering happens later via reprojection checks. This keeps tracks connected
// on low-parallax video-style input where per-pair essential matrices are
// ill-conditioned. Tracks chain matches across many frames, connecting wide
// baselines that direct matching cannot.
//
// All geometry runs on normalized image coordinates. Since EXIF is not read,
// several focal-length guesses are tried (relative to f = 1.2 * max(w, h)).

import { detectAndDescribe, detectAndDescribeMS } from './features.js';
import { detectSift, matchSift } from './sift.js';
import { gpuMatchAll } from './gpumatch.js';
import { matchDescriptors } from './matching.js';
import {
  I3, ransacE, decomposeE, selectPose, triangulateN, parallaxAngle,
  pnpRansac, refinePose, reprojError, makeRng, m3t, m3mulv,
  rodrigues, m3mul, solveLinear, p3pBearings,
} from './geometry.js';
import { bundleAdjust } from './ba.js';
import { rotationAveraging, globalPositionsJoint } from './global.js';

const MAXF = 32768; // feature-id stride per image (must exceed per-image feature count; SIFT emits up to 2x the
// requested budget). 16384 -> 32768 on 2026-09-08: a 16000-feature budget at featMaxDim 3200 (48 MP
// statue photos) tripped it. Only the union-find is sized n*MAXF (612 faces -> 20 M ids, fine on desktop).
                    // orientations/keypoint, so an 8000-keypoint budget from the upsampled octave needs > 8192 — garden
                    // hit 'too many features per image' at 8192, 2026-09-04; only the union-find stride depends on it)
// Focal-search candidates, relative to the 1.2*maxDim guess — anchored on
// real lenses since 2026-09-10 (f / long side = 0.029 x the 35 mm-equivalent
// focal for a 4:3 frame): 0.575 = 24 mm, the 1x main lens of every recent
// phone; 0.65 = 28 mm (older phones, video crops); 0.8 = 33 mm; 1.0 = 40 mm;
// 1.16 = 48 mm (phone 2x); 1.3 = 54 mm. The old grid started at 0.65 and a
// phone set's true focal sat 12 % below it, where BA did not follow. Beyond
// the edges (0.5x ultra-wide at 0.31, 3x tele at 1.85) the search brackets
// outward step by step while the data keeps improving. Sets with EXIF skip
// all of this (focalPrior).
const FOCAL_SCALES = [1.0, 0.8, 0.65, 0.575, 1.16, 1.3];

/** Solve-quality tiers — the camera solve's share of the quality presets
 *  (2026-09-09: a two-minute draft training sat behind a ten-minute solve).
 *  quick    = the pre-09-04 defaults: 3900 features, base octave, no aspect
 *             term (truck solve ~4 min on an RTX 5080)
 *  standard = the 8000-feature budget at the base octave, no aspect term
 *  precise  = 8000 features from the upsampled first octave + the pixel-
 *             aspect term in BA (the 09-04 desktop defaults: truck +0.38 dB
 *             at 30k, ~12-minute solve). The aspect term belongs with the
 *             fine features only (coarse + aspect drifted camping −0.6 dB).
 *  Rigs (sliced panoramas) clamp the octave and lock intrinsics regardless
 *  (session.solve). Explicit sfm options passed alongside always win. */
// MEASURED 2026-09-09 (truck-251, fresh solve + 30k, RTX 5080): quick 3.6 min
// → 25.51 dB; "8000 features at octave 0" 3.5 min → 25.42 (worse than 3900:
// the base octave has no 8000 good keypoints at 960 px) — withdrawn; the
// upsampled-octave recipe 10.8 min → 25.86, and 7.0 min → 26.03 once the
// init-pair fix, the worker RANSAC and the 10 k-point interim BA were in.
// So "standard" IS the precise recipe made cheaper; 'precise' stays as an
// alias for the bench and older links. (This block was lost once to a
// `git checkout -- src` on 2026-09-10 and restored the same day.)
export const SOLVE_TIERS = {
  quick:    { siftFeats: 3900, siftFirstOctave: 0,  refineAspect: false },
  standard: { siftFeats: 8000, siftFirstOctave: -1, refineAspect: true },
  precise:  { siftFeats: 8000, siftFirstOctave: -1, refineAspect: true },
};
export const solveTierOpts = (name) => ({ ...(SOLVE_TIERS[name] || SOLVE_TIERS.standard) });

const tick = () => new Promise((r) => setTimeout(r, 0));

class UnionFind {
  constructor(n) {
    this.p = new Int32Array(n);
    for (let i = 0; i < n; i++) this.p[i] = i;
  }
  find(x) {
    let r = x;
    while (this.p[r] !== r) r = this.p[r];
    while (this.p[x] !== r) { const nx = this.p[x]; this.p[x] = r; x = nx; }
    return r;
  }
  union(a, b) {
    const ra = this.find(a), rb = this.find(b);
    if (ra !== rb) this.p[ra] = rb;
  }
}

// Two graph profiles, declared per scene (index.json "graph"; default 'walk'):
//   walk  — neighbor window 10, sparse long-range. Safe for handheld paths:
//           on self-similar content (camping tents) wider windows admit
//           epipolar-consistent repeated-texture mismatches that poison the
//           union-find tracks and cost whole sections (113 -> 64 registered).
//   orbit — window 16 + dense long-range. Slow orbits (T&T train/truck)
//           genuinely overlap at those spans and the extra constraints stiffen
//           the chain: ATE 10% -> 4.5% on train-84.
function buildPairs(n, profile = 'walk') {
  const dense = profile === 'dense';
  const orbit = profile === 'orbit' || dense;
  const win = dense ? 20 : orbit ? 16 : 10;
  const pairs = [];
  if (n <= 30) {
    for (let i = 0; i < n; i++)
      for (let j = i + 1; j < n; j++) pairs.push([i, j]);
  } else {
    const seen = new Set();
    const add = (i, j) => {
      if (i === j || i < 0 || j >= n) return;
      const k = i < j ? i * 10000 + j : j * 10000 + i;
      if (!seen.has(k)) { seen.add(k); pairs.push(i < j ? [i, j] : [j, i]); }
    };
    for (let i = 0; i < n; i++)
      for (let d = 1; d <= win; d++) add(i, i + d);
    // long-range pairs for loop closure ('dense': every i, every 2nd j —
    // the subsampled grid left long capture loops pinned too loosely)
    for (let i = 0; i < n; i += dense ? 1 : orbit ? 2 : 4)
      for (let j = i + win + 2; j < n; j += dense ? 2 : orbit ? 3 : 4) add(i, j);
  }
  return pairs;
}

// ---- subpixel observation refinement (Lucas-Kanade) ----
// Corner detections are pixel-quantized (~1.5px BA residual floor); a chain
// of short-baseline cameras with that much obs noise has smooth low-frequency
// bend modes as its NOISE FLOOR — no amount of bundle adjustment removes them.
// Every pixel tolerance in this file was tuned on a 960 px feature frame (the
// phone/desktop default). A finer frame (opts.featMaxDim 2400/3200 for 48 MP
// photos) made the SAME pixel numbers 2.5-3.3x stricter in angle and
// registration collapsed (statue set 2026-09-08: 19/34 at 960 -> 7/34 at 2400,
// 8/34 at 3200 with 3-4 px rms). Scale them with the frame instead.
const pxScaleOf = (im) => Math.max(1, Math.max(im.fw, im.fh) / 960);

// Align each observation's patch against a single reference patch of its track
// (zero-mean SSD, translation-only LK) so all obs of a track agree at subpixel
// on the SAME physical point.
function refineObsLK(images, feats, tracks, poses, vlog) {
  const W = 5;                 // half window -> 11x11 patch
  const MAXSHIFT = 2.5 * pxScaleOf(images[0]); // px at the feature scale; larger moves are mismatches
  const P = 2 * W + 1;
  const refP = new Float32Array(P * P);
  const curP = new Float32Array(P * P);
  const gxP = new Float32Array(P * P);
  const gyP = new Float32Array(P * P);

  const bil = (g, w, h, x, y) => {
    const x0 = x | 0, y0 = y | 0;
    const fx = x - x0, fy = y - y0;
    const o = y0 * w + x0;
    return g[o] * (1 - fx) * (1 - fy) + g[o + 1] * fx * (1 - fy) +
           g[o + w] * (1 - fx) * fy + g[o + w + 1] * fx * fy;
  };

  let refined = 0, tried = 0, rejected = 0;
  const t0 = performance.now();
  for (const tr of tracks) {
    if (!tr.X) continue;
    const ok = tr.obs.filter((o) => o.ok && poses[o.img]);
    if (ok.length < 2) continue;
    const sorted = ok.slice().sort((a, b) => a.img - b.img);
    const ref = sorted[sorted.length >> 1];   // middle obs = reference
    const imR = images[ref.img];
    const rx = feats[ref.img].x[ref.feat], ry = feats[ref.img].y[ref.feat];
    if (rx < W + 2 || ry < W + 2 || rx > imR.fw - W - 3 || ry > imR.fh - W - 3) continue;
    let refMean = 0;
    for (let dy = -W, k = 0; dy <= W; dy++)
      for (let dx = -W; dx <= W; dx++, k++) refMean += refP[k] = bil(imR.gray, imR.fw, imR.fh, rx + dx, ry + dy);
    refMean /= P * P;

    for (const o of sorted) {
      if (o === ref) continue;
      tried++;
      const im = images[o.img];
      const x0 = feats[o.img].x[o.feat], y0 = feats[o.img].y[o.feat];
      let u = 0, v = 0, okConv = false;
      for (let it = 0; it < 10; it++) {
        const cx = x0 + u, cy = y0 + v;
        if (cx < W + 2 || cy < W + 2 || cx > im.fw - W - 3 || cy > im.fh - W - 3) break;
        let curMean = 0;
        for (let dy = -W, k = 0; dy <= W; dy++)
          for (let dx = -W; dx <= W; dx++, k++) {
            curP[k] = bil(im.gray, im.fw, im.fh, cx + dx, cy + dy);
            gxP[k] = 0.5 * (bil(im.gray, im.fw, im.fh, cx + dx + 1, cy + dy) - bil(im.gray, im.fw, im.fh, cx + dx - 1, cy + dy));
            gyP[k] = 0.5 * (bil(im.gray, im.fw, im.fh, cx + dx, cy + dy + 1) - bil(im.gray, im.fw, im.fh, cx + dx, cy + dy - 1));
            curMean += curP[k];
          }
        curMean /= P * P;
        let a00 = 0, a01 = 0, a11 = 0, b0 = 0, b1 = 0;
        for (let k = 0; k < P * P; k++) {
          const e = (curP[k] - curMean) - (refP[k] - refMean);
          a00 += gxP[k] * gxP[k]; a01 += gxP[k] * gyP[k]; a11 += gyP[k] * gyP[k];
          b0 -= gxP[k] * e; b1 -= gyP[k] * e;
        }
        const det = a00 * a11 - a01 * a01;
        if (det < 1e-12) break;
        const du = (a11 * b0 - a01 * b1) / det;
        const dv = (a00 * b1 - a01 * b0) / det;
        u += du; v += dv;
        if (u * u + v * v > MAXSHIFT * MAXSHIFT) { u = 1e9; break; }
        if (du * du + dv * dv < 1e-4) { okConv = true; break; }
      }
      if (okConv && u * u + v * v <= MAXSHIFT * MAXSHIFT) {
        feats[o.img].x[o.feat] = x0 + u;
        feats[o.img].y[o.feat] = y0 + v;
        refined++;
      } else rejected++;
    }
  }
  vlog(`subpixel LK refine: ${refined}/${tried} obs refined (${rejected} rejected) ` +
       `in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
}

// ---- guided track extension ----
// For every triangulated track, project its (bundle-adjusted) 3D point into
// each registered frame that does not observe it yet, and look for an
// unclaimed detected feature within a small pixel radius whose descriptor
// matches the track (min Hamming over the track's existing descriptors).
// Returns the list of added observations [{ tr, o }].
function extendTracks(feats, tracks, poses, K, regList, k1, k2, im0, vlog) {
  const DESC_WORDS = 8;
  const pxs = pxScaleOf(im0);
  const RADIUS = 4 * pxs;  // px search radius around the projection (960-px tuned)
  const HAM_MAX = 64;      // stricter than the 90 used for blind matching
  const MARGIN = 6 * pxs;  // stay away from the frame border
  const CELL = Math.ceil(8 * pxs); // spatial-grid cell size in px
  const w = im0.fw, h = im0.fh;

  const hamming = (dA, oA, dB, oB) => {
    let d = 0;
    for (let k = 0; k < DESC_WORDS; k++) {
      let v = (dA[oA + k] ^ dB[oB + k]) >>> 0;
      v = v - ((v >>> 1) & 0x55555555);
      v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
      d += (((v + (v >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
    }
    return d;
  };
  // SIFT descriptors are 128 x uint8 with L2 matching; binary BRIEF uses
  // Hamming. Pick per feature type.
  const isSift = !!feats[regList[0]].sift;
  const STRIDE = isSift ? 128 : DESC_WORDS;
  const DMAX = isSift ? 40000 : HAM_MAX; // L2^2 (rootSIFT x512) vs bits
  const descDist = isSift
    ? (dA, oA, dB, oB) => {
        let s = 0;
        for (let k = 0; k < 128; k++) {
          const d = dA[oA + k] - dB[oB + k];
          s += d * d;
          if (s > DMAX) return s;
        }
        return s;
      }
    : hamming;

  // spatial grid + claimed flags per registered image
  const gw = Math.ceil(w / CELL), gh = Math.ceil(h / CELL);
  const grid = new Map();     // img -> Map(cell -> [featIdx])
  const claimed = new Map();  // img -> Uint8Array(nFeats)
  for (const img of regList) {
    const f = feats[img];
    const g = new Map();
    for (let i = 0; i < f.n; i++) {
      const c = Math.min(gh - 1, Math.max(0, (f.y[i] / CELL) | 0)) * gw +
                Math.min(gw - 1, Math.max(0, (f.x[i] / CELL) | 0));
      let a = g.get(c);
      if (!a) g.set(c, a = []);
      a.push(i);
    }
    grid.set(img, g);
    claimed.set(img, new Uint8Array(f.n));
  }
  for (const tr of tracks)
    for (const o of tr.obs) {
      const cl = claimed.get(o.img);
      if (cl) cl[o.feat] = 1;
    }

  let lenBefore = 0, nTracks = 0;
  for (const tr of tracks) {
    if (!tr.X) continue;
    nTracks++;
    for (const o of tr.obs) if (o.ok && poses[o.img]) lenBefore++;
  }

  const extObs = [];
  for (const tr of tracks) {
    if (!tr.X) continue;
    const okObs = tr.obs.filter((o) => o.ok && poses[o.img]);
    if (okObs.length < 2) continue;
    const has = new Set(tr.obs.map((o) => o.img)); // incl. rejected obs
    // representative descriptors (cap for speed)
    const descs = okObs.slice(0, 6).map((o) => ({ d: feats[o.img].desc, off: o.feat * STRIDE }));
    for (const img of regList) {
      if (has.has(img)) continue;
      const R = poses[img].R, t = poses[img].t, X = tr.X;
      const pz = R[6]*X[0] + R[7]*X[1] + R[8]*X[2] + t[2];
      if (pz < 1e-9) continue;
      const x = (R[0]*X[0] + R[1]*X[1] + R[2]*X[2] + t[0]) / pz;
      const y = (R[3]*X[0] + R[4]*X[1] + R[5]*X[2] + t[1]) / pz;
      const r2 = x*x + y*y;
      const D = 1 + k1*r2 + k2*r2*r2;
      const u = K[img].f * x * D + K[img].cx;
      const v = K[img].f * y * D + K[img].cy;
      if (!(u > MARGIN && u < w - MARGIN && v > MARGIN && v < h - MARGIN)) continue;
      const f = feats[img], g = grid.get(img), cl = claimed.get(img);
      const cu = (u / CELL) | 0, cv = (v / CELL) | 0;
      let best = -1, bestHam = DMAX + 1, bestD2 = Infinity;
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++) {
          const cy2 = cv + dy, cx2 = cu + dx;
          if (cx2 < 0 || cy2 < 0 || cx2 >= gw || cy2 >= gh) continue;
          const cell = g.get(cy2 * gw + cx2);
          if (!cell) continue;
          for (const fi of cell) {
            if (cl[fi]) continue;
            const du = f.x[fi] - u, dv = f.y[fi] - v;
            const d2 = du*du + dv*dv;
            if (d2 > RADIUS * RADIUS) continue;
            let hm = Infinity;
            for (const dd of descs) hm = Math.min(hm, descDist(f.desc, fi * STRIDE, dd.d, dd.off));
            if (hm < bestHam || (hm === bestHam && d2 < bestD2)) { best = fi; bestHam = hm; bestD2 = d2; }
          }
        }
      if (best >= 0 && bestHam <= DMAX) {
        const o = { img, feat: best, ok: true };
        tr.obs.push(o);
        has.add(img);
        cl[best] = 1;
        extObs.push({ tr, o });
      }
    }
  }
  vlog(`track extension: +${extObs.length} obs ` +
       `(mean track length ${(lenBefore / Math.max(1, nTracks)).toFixed(2)} -> ` +
       `${((lenBefore + extObs.length) / Math.max(1, nTracks)).toFixed(2)})`);
  return extObs;
}

/**
 * Run incremental SfM.
 * images: [{ name, fw, fh, gray: Float32Array }]
 * log: (msg) => void
 * sampleColor: (imgIdx, x, y) => [r, g, b]
 * Returns { cams, points }.
 */
async function runSfMOnce(images, log, sampleColor, opts = {}) {
  const n = images.length;
  if (n < 2) throw new Error('need at least 2 images');
  const signal = opts.signal;
  const checkAbort = () => {
    if (signal && signal.aborted) throw new DOMException('SfM aborted', 'AbortError');
  };
  // structured progress alongside log(): { stage, done, total, detail }
  // stages: 'features' | 'matching' | 'focal' | 'register' | 'ba'
  const ev = opts.onEvent || (() => {});

  // ---- features ----
  // SIFT is the DEFAULT since 2026-08-19: it beat or tied oriented-BRIEF on
  // every validation scene (truck/train-84 ATE 0.04% = COLMAP parity, camping
  // 27.2, synthetic 43.0/40.8) with worker extraction + GPU matching keeping
  // it fast. features:'brief' restores the old binary pipeline.
  const useSift = opts.features !== 'brief';
  // the landmark budget scales with feature AREA — a fixed 3900 spread over
  // 1600px images is 2.8x sparser than at 960 and the weakened match chains
  // DOUBLED camping's trajectory error. 3900 at 960px is COLMAP-parity
  // density; the 8192 ceiling is COLMAP's own default cap (and our MAXF).
  // An explicit opts.siftFeats always wins.
  if (!opts.siftFeats) {
    const featDim = Math.max(1, ...images.map((im) => Math.max(im.fw || 0, im.fh || 0)));
    const scaled = Math.round(3900 * Math.pow(featDim / 960, 2));
    opts = { ...opts, siftFeats: Math.max(3900, Math.min(8192, scaled)) };
    if (opts.siftFeats > 3900) log(`feature budget ${opts.siftFeats} (density-scaled for ${featDim}px)`);
  }
  const feats = [];
  if (opts._feats) {
    for (const f of opts._feats) feats.push(f);
    log(`features: reused from the first pass (${n} images)`);
  } else {
  log(`detecting features in ${n} images ...`);
  if (useSift && typeof Worker !== 'undefined' && opts.workers !== false) {
    // SIFT extraction is ~1s/image of pure CPU — run it on a worker pool
    const t0f = performance.now();
    const nW = Math.min(opts.workers || 8, Math.max(2, (navigator.hardwareConcurrency || 4) - 2));
    const workers = Array.from({ length: nW },
      () => new Worker(new URL('./featworker.js', import.meta.url), { type: 'module' }));
    const results = new Array(n);
    let doneF = 0;
    await new Promise((resolve, reject) => {
      let next = 0;
      const feed = (wk) => {
        if (next >= n) return;
        const id = next++;
        wk.onmessage = (e) => {
          results[e.data.id] = e.data;
          doneF++;
          // detail carries the REAL keypoints (references, feature scale) so a
          // UI can draw them as they arrive
          ev({ stage: 'features', done: doneF, total: n,
               detail: { image: e.data.id, n: e.data.n, x: e.data.x, y: e.data.y } });
          if (doneF % 16 === 0) log(`  features: ${doneF}/${n}`);
          if (doneF === n) resolve();
          else feed(wk);
        };
        wk.onerror = (err) => reject(new Error('feature worker: ' + (err.message || err)));
        // fine-scale defaults: firstOctave -1 (upsampled octave) flooded the
        // contrast-sorted cap with poorly-localized features and warped truck
        // to 8.5% ATE; firstOctave 0 + 1800 feats = 0.04% on truck AND train
        wk.postMessage({
          id, gray: images[id].gray, w: images[id].fw, h: images[id].fh,
          maxFeats: opts.siftFeats || 3900, firstOctave: opts.siftFirstOctave ?? 0,
          peakScale: opts.siftPeak ?? 1,
        });
      };
      workers.forEach(feed);
    });
    workers.forEach((w) => w.terminate());
    for (let i = 0; i < n; i++) {
      const f = results[i];
      f.sift = true;
      if (f.n > MAXF) throw new Error('too many features per image');
      f.xn = new Float32Array(f.n);
      f.yn = new Float32Array(f.n);
      feats.push(f);
    }
    log(`  SIFT extraction: ${n} images on ${nW} workers in ${((performance.now() - t0f) / 1000).toFixed(1)}s`);
  } else {
    for (let i = 0; i < n; i++) {
      // 1500 is deliberate: raising it (tried 2200) admits weaker corners whose
      // BRIEF descriptors mismatch, and pose accuracy measurably degrades.
      // features:'sift' = real scale-space SIFT (COLMAP-grade, validated at
      // 82.5% keypoint recall vs COLMAP's own extraction; slower).
      const f = useSift
        ? detectSift(images[i].gray, images[i].fw, images[i].fh, opts.siftFeats || 3900,
            opts.siftFirstOctave ?? 0, opts.siftPeak ?? 1)
        : opts.msFeatures
          ? detectAndDescribeMS(images[i].gray, images[i].fw, images[i].fh, 1500)
          : detectAndDescribe(images[i].gray, images[i].fw, images[i].fh, 1500);
      f.sift = useSift;
      if (f.n > MAXF) throw new Error('too many features per image');
      f.xn = new Float32Array(f.n);
      f.yn = new Float32Array(f.n);
      feats.push(f);
      ev({ stage: 'features', done: i + 1, total: n });
      if (i % 8 === 7) { log(`  features: ${i + 1}/${n}`); await tick(); checkAbort(); }
    }
  }
  checkAbort();
  log(`  avg features/image: ${Math.round(feats.reduce((s, f) => s + f.n, 0) / n)}`);
  await tick();

  // Adaptive rescue: images that came back feature-poor get a second pass
  // with the base octave upsampled 2x (COLMAP's own default). Soft indoor
  // texture — painted walls, ceilings — is invisible at octave 0 and those
  // frames starve registration (playroom: 150/225 -> 209/225). ONLY the
  // starved images: fine-scale features on rich images once flooded the
  // contrast-sorted cap and warped truck to 8.5% ATE (note above).
  if (useSift && (opts.siftFirstOctave ?? 0) >= 0) {
    // density, not an absolute count: fewer than one keypoint per ~700
    // feature-scale pixels marks a starved image. An absolute threshold
    // dragged camping's small video frames into the rescue and the
    // fine-scale flood cost it 2.15% ATE (0.38% at 1/500; 1/700 leaves its
    // marginal frames alone while playroom's blank walls still qualify).
    const starved = [];
    for (let i = 0; i < n; i++) {
      if (feats[i].n * 700 < images[i].fw * images[i].fh) starved.push(i);
    }
    if (starved.length) {
      log(`  ${starved.length} feature-poor images (density) — second pass at first octave -1`);
      const t0r = performance.now();
      const adopt = (data) => {
        const f = data;
        f.sift = true;
        if (f.n > MAXF) throw new Error('too many features per image');
        f.xn = new Float32Array(f.n);
        f.yn = new Float32Array(f.n);
        feats[f.id] = f;
      };
      if (typeof Worker !== 'undefined' && opts.workers !== false) {
        const nW = Math.min(opts.workers || 8, Math.max(2, (navigator.hardwareConcurrency || 4) - 2));
        const workers = Array.from({ length: nW },
          () => new Worker(new URL('./featworker.js', import.meta.url), { type: 'module' }));
        let doneR = 0;
        await new Promise((resolve, reject) => {
          let next = 0;
          const feed = (wk) => {
            if (next >= starved.length) return;
            const id = starved[next++];
            wk.onmessage = (e) => {
              adopt(e.data);
              doneR++;
              ev({ stage: 'features', done: doneR, total: starved.length,
                   detail: { image: e.data.id, n: e.data.n, x: e.data.x, y: e.data.y } });
              if (doneR === starved.length) resolve();
              else feed(wk);
            };
            wk.onerror = (err) => reject(new Error('feature worker: ' + (err.message || err)));
            wk.postMessage({
              id, gray: images[id].gray, w: images[id].fw, h: images[id].fh,
              maxFeats: opts.siftFeats || 3900, firstOctave: -1,
              peakScale: opts.siftPeak ?? 1,
            });
          };
          workers.forEach(feed);
        });
        workers.forEach((w) => w.terminate());
      } else {
        for (const id of starved) {
          const f = detectSift(images[id].gray, images[id].fw, images[id].fh,
            opts.siftFeats || 3900, -1, opts.siftPeak ?? 1);
          f.id = id;
          adopt(f);
          await tick(); checkAbort();
        }
      }
      const avg2 = Math.round(starved.reduce((s, id) => s + feats[id].n, 0) / starved.length);
      log(`  rescue: avg ${avg2} features on those images ` +
        `(${((performance.now() - t0r) / 1000).toFixed(1)}s)`);
      checkAbort();
      await tick();
    }
  }

  }   // end of feature extraction (skipped when opts._feats is cached)

  // ---- pairwise matching ----
  // Mutual matches build the track graph. When an essential matrix fits a
  // pair well, its inliers replace the raw matches (removes contamination on
  // wide-baseline pairs); when E is unreliable (low-parallax video pairs) the
  // raw matches are kept so tracks stay connected.
  const matchRng = makeRng(24680);
  const K0 = images.map((im) => ({
    f: 1.2 * Math.max(im.fw, im.fh), cx: im.fw / 2, cy: im.fh / 2,
  }));
  // With SIFT the wide 'orbit' graph is safe everywhere (the walk-window
  // limit was a BRIEF pathology: repeated-texture mismatches at wide
  // baselines poisoned tracks — camping went 113 -> 64 registered). SIFT
  // camping with the wide graph: 113/113, best-ever PSNR, tail drift down.
  // 'dense' since 2026-08-21: the subsampled long-range grid left a 250-image
  // loop pinned so loosely that two 8-16 camera clusters registered ~5 units
  // off (a ghost truck in training). Dense long-range: truck-250 ATE
  // 2.18% -> 0.00% vs COLMAP. GPU matching absorbs the extra pairs.
  const pairs = buildPairs(n, opts.graph || (useSift ? 'dense' : 'walk'));
  log(`matching ${pairs.length} image pairs ...`);
  const t0m = performance.now();
  const pairInfo = []; // { i, j, matches: [[fa, fb], ...] }
  const failedRich = []; // many matches but failed the E-gate (rescue candidates)
  let done = 0, filtered = 0;
  // per-image diagnostics for the log: usable-pair degree, best raw match count,
  // best E-inlier count — the answer to "why did these frames not register"
  const diag = images.map(() => ({ deg: 0, raw: 0, inl: 0 }));
  const adjFails = [];   // rejected neighbour pairs (|i-j| <= 2), logged for the connectivity verdict
  const pairDbg = [];    // opts.pairDebug: every neighbour pair (raw/E-inl/hash)
  // SIFT matching runs on GPU when available (CPU 128-D L2 is the dominant
  // cost of a SIFT run: ~350s of 484s on train-84; the GPU does the whole
  // graph in ~1s of compute)
  let gpuAll = null;
  if (useSift && typeof navigator !== 'undefined' && navigator.gpu) {
    try {
      gpuAll = await gpuMatchAll(feats, pairs, 0.8, log, opts.gpu && opts.gpu.device,
        (d, t) => ev({ stage: 'matching', done: d, total: t, detail: { usable: 0, pair: null, gpu: true } }));
    } catch (e) {
      log(`  GPU matcher unavailable (${e.message}) — CPU fallback`);
    }
  }
  // (1) mutual matches per pair + the normalised coordinates the E-check needs
  const jobs = [];          // { pi, i, j, matches, x1, x2, thresh }
  const rawMatches = new Array(pairs.length);
  for (let pi = 0; pi < pairs.length; pi++) {
    const [i, j] = pairs[pi];
    const m = gpuAll
      ? gpuAll[pi]
      : feats[i].sift
        ? matchSift(feats[i].desc, feats[i].n, feats[j].desc, feats[j].n)
        : matchDescriptors(feats[i].desc, feats[i].n, feats[j].desc, feats[j].n);
    diag[i].raw = Math.max(diag[i].raw, m.length / 2); diag[j].raw = Math.max(diag[j].raw, m.length / 2);
    if (m.length / 2 < 25) continue;
    const matches = [];
    for (let k = 0; k < m.length; k += 2) matches.push([m[k], m[k + 1]]);
    const x1 = new Float32Array(matches.length * 2), x2 = new Float32Array(matches.length * 2);
    for (let k = 0; k < matches.length; k++) {
      const [fa, fb] = matches[k];
      x1[2 * k] = (feats[i].x[fa] - K0[i].cx) / K0[i].f; x1[2 * k + 1] = (feats[i].y[fa] - K0[i].cy) / K0[i].f;
      x2[2 * k] = (feats[j].x[fb] - K0[j].cx) / K0[j].f; x2[2 * k + 1] = (feats[j].y[fb] - K0[j].cy) / K0[j].f;
    }
    const favg = (K0[i].f + K0[j].f) / 2;
    jobs.push({ pi, i, j, matches, x1, x2, thresh: 2.5 / favg });
    rawMatches[pi] = matches;
    if (!gpuAll && pi % 40 === 39) { await tick(); checkAbort(); }
  }

  // (2) essential-matrix RANSAC per pair — independent work, so a worker pool
  // (src/sfm/pairworker.js) does it in parallel; the main thread only gates.
  // Per-pair seeds keep the result independent of the batch split. Without
  // workers (tests, opts.workers === false) the loop runs inline as before.
  const inliersOf = new Array(pairs.length).fill(null);
  const statsOf = new Array(pairs.length).fill(null);   // opts.pairDebug: RANSAC stats per pair
  const useWorkers = typeof Worker !== 'undefined' && opts.workers !== false && opts.pairWorkers !== false && jobs.length > 32;
  if (useWorkers) {
    const nW = Math.min(opts.workers || 8, Math.max(2, (navigator.hardwareConcurrency || 4) - 2));
    const workers = Array.from({ length: nW },
      () => new Worker(new URL('./pairworker.js', import.meta.url), { type: 'module' }));
    const BATCH = 24;
    let next = 0, doneJobs = 0;
    const seedOf = (pi) => 24680 + pi * 7919;
    await new Promise((resolve, reject) => {
      const feed = (wk) => {
        if (next >= jobs.length) return;
        const slice = jobs.slice(next, next + BATCH);
        next += slice.length;
        wk.onmessage = (e) => {
          for (const r of e.data.out) { inliersOf[r.idx] = r.inliers; if (r.stats) statsOf[r.idx] = r.stats; }
          doneJobs += e.data.out.length;
          ev({ stage: 'matching', done: Math.round(pairs.length * doneJobs / jobs.length), total: pairs.length,
               detail: { usable: 0, pair: null } });
          if (doneJobs >= jobs.length) resolve();
          else feed(wk);
        };
        wk.onerror = (err) => reject(new Error('pair worker: ' + (err.message || err)));
        wk.postMessage({ batch: slice.map((jb) => ({ idx: jb.pi, x1: jb.x1, x2: jb.x2, thresh: jb.thresh, seed: seedOf(jb.pi), maxIters: opts.pairIters || 600, debug: !!opts.pairDebug })) });
      };
      workers.forEach(feed);
    });
    workers.forEach((w) => w.terminate());
  } else {
    for (let q = 0; q < jobs.length; q++) {
      const jb = jobs[q];
      const m = jb.matches.length;
      const x1s = new Array(m), x2s = new Array(m);
      for (let k = 0; k < m; k++) { x1s[k] = [jb.x1[2 * k], jb.x1[2 * k + 1]]; x2s[k] = [jb.x2[2 * k], jb.x2[2 * k + 1]]; }
      const res = ransacE(x1s, x2s, jb.thresh, matchRng, opts.pairIters || 600);
      inliersOf[jb.pi] = res ? Int32Array.from(res.inliers) : null;
      if (q % 40 === 39) { await tick(); checkAbort(); }
    }
  }

  // (3) the gate, in pair order (deterministic bookkeeping)
  for (let pi = 0; pi < pairs.length; pi++) {
    const [i, j] = pairs[pi];
    done++;
    let matches = rawMatches[pi];
    if (matches) {
      const inl = inliersOf[pi];
      const res = inl ? { inliers: inl } : null;
      if (res) { diag[i].inl = Math.max(diag[i].inl, res.inliers.length); diag[j].inl = Math.max(diag[j].inl, res.inliers.length); }
      // pairs whose matches can't support an essential matrix are junk —
      // letting their raw matches into the track graph merges unrelated
      // tracks, which then get dropped as conflicted
      // Accept on the inlier RATIO (>= 40 %: the pair is trustworthy) OR on an absolute
      // inlier count. MEASURED 2026-09-08 on a low-texture statue: raw matches inflate
      // with false matches on repetitive surfaces, so a pair with 304 E-inliers of 2057
      // raw (15 %) was thrown away and finer feature frames registered FEWER images
      // (19/34 at 960 px -> 7/34 at 2400). opts.pairMinInliers (default 100) is the
      // absolute gate; the E-filtered inliers are all that enter the track graph.
      // Neighbours in capture order (|i-j| <= 2: photo sets arrive time-sorted, video
      // frames sequential) get a lower absolute gate (opts.pairMinInliersAdj, default
      // 15): a walk-around's consecutive photos overlap by construction, and one lost
      // link there strands the rest of the arc.
      // OPT-IN (opts.pairMinInliersAdj): a truck fresh solve with the neighbour gate at 15
      // registered 250/251 at rms 0.62 yet trained to 22.67 dB (25.72 without it) — a
      // low-inlier neighbour pair can be geometrically wrong and poison the graph. Never
      // for faces of the same rig (zero baseline). Default off until the failure is understood.
      const sameRig = opts.rigs && opts.rigs[i] && opts.rigs[j] && opts.rigs[i].id === opts.rigs[j].id;
      const nb = !sameRig && Math.abs(i - j) <= 2;          // neighbours in capture order (logged when rejected)
      if (opts.pairDebug && nb) {   // diagnostics (2026-09-17): is the pair stage deterministic? raw count, E-inliers, a hash of the raw matches
        let h = 2166136261; for (const [a, b] of matches) { h = Math.imul(h ^ a, 16777619); h = Math.imul(h ^ b, 16777619); }
        // median pixel displacement of the raw matches (a near-zero baseline or a duplicate frame shows here)
        const fav = (K0[i].f + K0[j].f) / 2, dd = matches.map(([fa, fb]) => Math.hypot(feats[i].x[fa] - feats[j].x[fb], feats[i].y[fa] - feats[j].y[fb])).sort((a, b) => a - b);
        const st = statsOf[pi]; const ex = st ? ` bc${st.bestCount} ef${st.estFails}/${st.iters}` : '';
        pairDbg.push(`${i}-${j}:${matches.length}/${res ? res.inliers.length : 0}/${(h >>> 0).toString(16).slice(0, 6)} d${dd[dd.length >> 1].toFixed(0)}px${ex}`);
      }
      const adj = nb && opts.pairMinInliersAdj > 0;
      const absGate = res && res.inliers.length >= (adj ? opts.pairMinInliersAdj : (opts.pairMinInliers ?? Infinity));
      if (res && res.inliers.length >= (adj ? 15 : 25) && (res.inliers.length >= 0.4 * matches.length || absGate)) {
        matches = Array.from(res.inliers, (idx) => matches[idx]);
        filtered++;
        diag[i].deg++; diag[j].deg++;
        pairInfo.push({ i, j, matches });
      } else {
        if (nb) adjFails.push(`${i}-${j}: raw ${matches.length}, E-inl ${res ? res.inliers.length : 0}`);
        if (matches.length >= 60) failedRich.push({ i, j, matches });
      }
    }
    if (pi % 40 === 39 || pi === pairs.length - 1) {
      const lastPair = pairInfo[pairInfo.length - 1];
      ev({ stage: 'matching', done, total: pairs.length, detail: {
        usable: pairInfo.length,
        // a drawable sample of the latest surviving pair (feature indices)
        pair: lastPair ? { i: lastPair.i, j: lastPair.j, sample: lastPair.matches.slice(0, 70) } : null,
      } });
      if (done % 200 === 0) log(`  pairs: ${done}/${pairs.length} (${pairInfo.length} usable)`);
      await tick(); checkAbort();
    }
  }
  checkAbort();
  log(`  usable pairs: ${pairInfo.length} (${filtered} E-filtered, rest raw) — ` +
      `matching + pair geometry ${((performance.now() - t0m) / 1000).toFixed(1)}s total`);
  {
    // isolated / weakly connected frames, with what their best pair looked like
    const weak = diag.map((d, k) => ({ k, ...d })).filter((d) => d.deg <= 1);
    if (weak.length) log(`  weakly connected frames (usable pairs <= 1): ` + weak.map((d) => `#${d.k}[deg ${d.deg}, best raw ${d.raw}, best E-inl ${d.inl}, feats ${feats[d.k].n}]`).join(' '));
    if (adjFails.length) log(`  rejected neighbour pairs: ` + adjFails.join(' | '));
    if (pairDbg.length) log(`  pairdebug: ` + pairDbg.join(' '));
  }
  if (pairInfo.length === 0) throw new Error('no image pair with enough matches');
  await tick();

  // ---- component rescue ----
  // The strict 40%-ratio E-gate keeps the dense track graph clean, but a
  // capture-session seam (truck frames 51|52) can leave the IMAGE graph split
  // with every bridging pair sitting just under the gate — losing one pair
  // then costs 32 of 84 frames. Retry only the match-rich failed pairs that
  // would connect two components, with a bigger RANSAC budget and an
  // absolute-consensus acceptance. In-component pairs never get the looser
  // rule (a global relaxation measurably poisons the graph: 49/84 registered,
  // spurious k1 0.22).
  if (failedRich.length) {
    const comp = new Int32Array(n);
    for (let i = 0; i < n; i++) comp[i] = i;
    const findC = (x) => {
      let r = x;
      while (comp[r] !== r) r = comp[r];
      while (comp[x] !== r) { const nx = comp[x]; comp[x] = r; x = nx; }
      return r;
    };
    for (const p of pairInfo) { const a = findC(p.i), b = findC(p.j); if (a !== b) comp[a] = b; }
    const pre = Array.from({ length: n }, (_, i) => findC(i)); // snapshot BEFORE rescue
    // only bridge SUBSTANTIAL components (two solid capture sessions, like
    // truck 52|32). Wide-baseline photo sets fragment into confetti — gluing
    // those with marginal geometry produces flipped cameras (playroom went
    // 7 -> 15 "registered" with visibly wrong poses before this guard).
    const compSize = new Map();
    for (const r of pre) compSize.set(r, (compSize.get(r) || 0) + 1);
    const cand = failedRich
      .filter((c) => pre[c.i] !== pre[c.j] &&
                     compSize.get(pre[c.i]) >= 6 && compSize.get(pre[c.j]) >= 6)
      .sort((a, b) => b.matches.length - a.matches.length)
      .slice(0, 20);
    let rescued = 0;
    for (const c of cand) {
      const x1s = c.matches.map(([fa]) => [(feats[c.i].x[fa] - K0[c.i].cx) / K0[c.i].f, (feats[c.i].y[fa] - K0[c.i].cy) / K0[c.i].f]);
      const x2s = c.matches.map(([, fb]) => [(feats[c.j].x[fb] - K0[c.j].cx) / K0[c.j].f, (feats[c.j].y[fb] - K0[c.j].cy) / K0[c.j].f]);
      const favg = (K0[c.i].f + K0[c.j].f) / 2;
      const res = ransacE(x1s, x2s, 2.5 / favg, matchRng, 3000);
      if (res && res.inliers.length >= 30 && res.inliers.length >= 0.15 * c.matches.length) {
        pairInfo.push({ i: c.i, j: c.j, matches: res.inliers.map((idx) => c.matches[idx]) });
        rescued++;
        log(`  component rescue: pair ${c.i}-${c.j} accepted (${res.inliers.length}/${c.matches.length} inliers)`);
      }
    }
    if (rescued) log(`  component rescue: +${rescued} bridging pairs`);
    await tick();
  }

  // ---- feature tracks (union-find over raw matches) ----
  const uf = new UnionFind(n * MAXF);
  for (const p of pairInfo)
    for (const [fa, fb] of p.matches)
      uf.union(p.i * MAXF + fa, p.j * MAXF + fb);

  const groups = new Map();
  for (const p of pairInfo) {
    for (const [fa, fb] of p.matches) {
      for (const id of [p.i * MAXF + fa, p.j * MAXF + fb]) {
        const root = uf.find(id);
        if (!groups.has(root)) groups.set(root, new Set());
        groups.get(root).add(id);
      }
    }
  }
  // tracks: keep only unambiguous observations (one feature per image)
  const tracks = [];
  const featTrack = images.map((im, i) => new Int32Array(feats[i].n).fill(-1));
  for (const g of groups.values()) {
    if (g.size < 2) continue;
    const perImg = new Map();
    for (const id of g) {
      const img = (id / MAXF) | 0;
      if (!perImg.has(img)) perImg.set(img, []);
      perImg.get(img).push(id % MAXF);
    }
    // a track with two features in one image is a bad union-find merge —
    // drop it entirely rather than salvaging (mixed tracks poison PnP)
    let conflicted = false;
    const obs = [];
    for (const [img, fs] of perImg) {
      if (fs.length > 1) { conflicted = true; break; }
      obs.push({ img, feat: fs[0], ok: true });
    }
    if (conflicted || obs.length < 2) continue;
    const tid = tracks.length;
    tracks.push({ obs, X: null });
    for (const o of obs) featTrack[o.img][o.feat] = tid;
  }
  log(`  tracks: ${tracks.length}`);
  await tick();

  // shared-track counts per image pair (focal-independent)
  const sharedCount = new Map();
  for (const tr of tracks) {
    const imgs = tr.obs.map((o) => o.img).sort((a, b) => a - b);
    for (let a = 0; a < imgs.length; a++)
      for (let b = a + 1; b < imgs.length; b++) {
        const k = imgs[a] * 10000 + imgs[b];
        sharedCount.set(k, (sharedCount.get(k) || 0) + 1);
      }
  }

  // =========================================================================
  // Geometry stage, parameterized by focal scale. Fully self-contained so the
  // focal search can run it repeatedly on reset track state.
  // =========================================================================
  // allowed: an optional Set of image indices — the focal search registers a
  // subsample (every k-th photo, <= 48) since it only ranks focals; the final
  // pass registers everything
  async function runGeometry(fScale, verbose, withBA = false, initSkip = 0, allowed = null) {
    const vlog = verbose ? log : () => {};
    const rng = makeRng(1234567);
    const K = images.map((im) => ({
      f: fScale * 1.2 * Math.max(im.fw, im.fh),
      cx: im.fw / 2,
      cy: im.fh / 2,
    }));
    for (let i = 0; i < n; i++) {
      const f = feats[i];
      // every pass starts from the DETECTED keypoints. The sub-pixel LK refinement of a final
      // pass writes its moved coordinates into feats[].x/y (refineObsLK), so a second final
      // pass — a trial, the retry — used to start from observations already pulled toward the
      // previous pass's geometry: the same focal that solved to 0.58 px as a first pass solved
      // to 0.64 px as a second, with the orbit's tail 1 % off (2026-09-17e). Snapshot once,
      // restore per pass; the kept pass's own refinement is what its poses were fitted to.
      if (!f.x0) { f.x0 = Float32Array.from(f.x); f.y0 = Float32Array.from(f.y); }
      else { f.x.set(f.x0); f.y.set(f.y0); }
      for (let k = 0; k < f.n; k++) {
        f.xn[k] = (f.x[k] - K[i].cx) / K[i].f;
        f.yn[k] = (f.y[k] - K[i].cy) / K[i].f;
      }
    }
    // reset track state — including the observations a previous final pass ADDED (extendTracks
    // pushes onto tr.obs under that pass's geometry; a later pass inherited them as evidence).
    // The first pass records each track's detected length; every pass truncates back to it.
    for (const tr of tracks) {
      if (tr.n0 == null) tr.n0 = tr.obs.length; else if (tr.obs.length > tr.n0) tr.obs.length = tr.n0;
      tr.X = null; for (const o of tr.obs) o.ok = true;
    }

    const obsNorm = (o) => [feats[o.img].xn[o.feat], feats[o.img].yn[o.feat]];
    const poses = new Array(n).fill(null);
    const registered = new Set();
    // a new registration pass in its own world frame (each focal candidate, the final run, the
    // bracket re-check, the gap retry): the app's overlay clears its frustum ring on this
    // final: the pass whose poses become the solve (withBA) — the focal-search candidates and the
    // init trials are not, and the app's overlay ignores their registrations
    ev({ stage: 'pass', done: 0, total: n, detail: { fScale: fScale * 1.2, final: !!withBA } });
    const failed = new Set();
    // Generous reprojection budget (feature-scale px): triangulation from
    // low-parallax pairs is noisy; global refinement tightens it afterwards.
    const thN = (img) => 6.0 * pxScaleOf(images[img]) / K[img].f;   // 6 px at 960, scaled with the frame

    function triangulateTrack(tr) {
      for (let attempt = 0; attempt < 2; attempt++) {
        const cams = [], obsPts = [], used = [];
        for (const o of tr.obs) {
          if (!o.ok || !poses[o.img]) continue;
          cams.push(poses[o.img]);
          obsPts.push(obsNorm(o));
          used.push(o);
        }
        if (cams.length < 2) { tr.X = null; return false; }
        const X = triangulateN(cams, obsPts);
        if (!X) { tr.X = null; return false; }
        let anyBad = false;
        for (let k = 0; k < cams.length; k++) {
          const err = reprojError(cams[k].R, cams[k].t, X, obsPts[k]);
          if (!(err < thN(used[k].img))) { used[k].ok = false; anyBad = true; }
        }
        if (anyBad) { tr.X = null; continue; }
        let maxAng = 0;
        for (let a = 0; a < cams.length - 1 && maxAng < 0.005; a++)
          for (let b = a + 1; b < cams.length && maxAng < 0.005; b++)
            maxAng = Math.max(maxAng, parallaxAngle(cams[a], cams[b], X));
        if (maxAng < 0.005) { tr.X = null; return false; }
        tr.X = X;
        return true;
      }
      return false;
    }

    const sfmOpts = opts;
    const useGlobal = !!opts.globalInit;

    // camera rigs (360 panos sliced into cubemap faces): opts.rigs[i] =
    // { id, R } with R the fixed rig->face rotation. All faces of a rig
    // share one centre, so registering ANY face poses the whole rig.
    const rigOf = opts.rigs || null;
    const mul3 = (A, B) => [
      A[0] * B[0] + A[1] * B[3] + A[2] * B[6], A[0] * B[1] + A[1] * B[4] + A[2] * B[7], A[0] * B[2] + A[1] * B[5] + A[2] * B[8],
      A[3] * B[0] + A[4] * B[3] + A[5] * B[6], A[3] * B[1] + A[4] * B[4] + A[5] * B[7], A[3] * B[2] + A[4] * B[5] + A[5] * B[8],
      A[6] * B[0] + A[7] * B[3] + A[8] * B[6], A[6] * B[1] + A[7] * B[4] + A[8] * B[7], A[6] * B[2] + A[7] * B[5] + A[8] * B[8],
    ];
    const tmul3 = (A, B) => mul3([A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]], B);
    const centreOf = (R, t) => [
      -(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]),
      -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]),
      -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2]),
    ];

    /** The rig's 2D-3D correspondences over ALL its faces, as bearing vectors
     *  in the RIG frame — one face is a narrow ambiguous view, the union
     *  looks in every direction (a central omnidirectional camera). */
    function corrsForRig(rid) {
      const obj = [], brg = [], face = [];
      for (let i = 0; i < n; i++) {
        if (!rigOf[i] || rigOf[i].id !== rid) continue;
        const Rf = rigOf[i].R;
        const ft = featTrack[i];
        for (let f2 = 0; f2 < feats[i].n; f2++) {
          const tid = ft[f2];
          if (tid < 0) continue;
          const tr = tracks[tid];
          if (!tr.X) continue;
          const o = tr.obs.find((oo) => oo.img === i);
          if (!o || !o.ok) continue;
          const xn = feats[i].xn[f2], yn = feats[i].yn[f2];
          const inv = 1 / Math.hypot(xn, yn, 1);
          // face ray -> rig frame (Rf is rig->face, so apply its transpose)
          brg.push([
            (Rf[0] * xn + Rf[3] * yn + Rf[6]) * inv,
            (Rf[1] * xn + Rf[4] * yn + Rf[7]) * inv,
            (Rf[2] * xn + Rf[5] * yn + Rf[8]) * inv,
          ]);
          obj.push(tr.X);
          face.push(i);
        }
      }
      return { obj, brg, face };
    }

    /** 6-DOF rig pose refinement on bearing vectors: Gauss-Newton with
     *  numeric jacobians on angular residuals, alternated with inlier
     *  re-selection (same shape as registerImage's tryFrom, but over the
     *  whole sphere at once). Returns { R, C, inliers } or null. */
    function refineRigAngular(R0, C0, obj, brg, thAng) {
      let R = Array.from(R0), C = C0.slice();
      const resid = (R2, C2, o, b) => {
        const dx = o[0] - C2[0], dy = o[1] - C2[1], dz = o[2] - C2[2];
        let vx = R2[0] * dx + R2[1] * dy + R2[2] * dz;
        let vy = R2[3] * dx + R2[4] * dy + R2[5] * dz;
        let vz = R2[6] * dx + R2[7] * dy + R2[8] * dz;
        const inv = 1 / (Math.hypot(vx, vy, vz) || 1e-12);
        return [vx * inv - b[0], vy * inv - b[1], vz * inv - b[2]];
      };
      const angErr = (R2, C2, o, b) => {
        const e = resid(R2, C2, o, b);
        return Math.hypot(e[0], e[1], e[2]);   // ~angle in radians (small)
      };
      let inliers = 0;
      for (let round = 0; round < 3; round++) {
        const io = [], ib = [];
        for (let k = 0; k < obj.length; k++)
          if (angErr(R, C, obj[k], brg[k]) < thAng * 1.5) { io.push(obj[k]); ib.push(brg[k]); }
        inliers = io.length;
        if (inliers < 12) return null;
        for (let it = 0; it < 5; it++) {
          const JTJ = new Float64Array(36), JTr = new Float64Array(6);
          const eps = 1e-5;
          const apply = (d) => {
            const Rn = m3mul(rodrigues([d[0], d[1], d[2]]), R);
            return [Rn, [C[0] + d[3], C[1] + d[4], C[2] + d[5]]];
          };
          for (let k = 0; k < io.length; k++) {
            const r0 = resid(R, C, io[k], ib[k]);
            const Jk = [];
            for (let p = 0; p < 6; p++) {
              const d = [0, 0, 0, 0, 0, 0];
              d[p] = eps;
              const [Rp, Cp] = apply(d);
              const rp = resid(Rp, Cp, io[k], ib[k]);
              Jk.push([(rp[0] - r0[0]) / eps, (rp[1] - r0[1]) / eps, (rp[2] - r0[2]) / eps]);
            }
            for (let a = 0; a < 6; a++) {
              for (let b2 = a; b2 < 6; b2++) {
                const v = Jk[a][0] * Jk[b2][0] + Jk[a][1] * Jk[b2][1] + Jk[a][2] * Jk[b2][2];
                JTJ[a * 6 + b2] += v;
                if (a !== b2) JTJ[b2 * 6 + a] += v;
              }
              JTr[a] += Jk[a][0] * r0[0] + Jk[a][1] * r0[1] + Jk[a][2] * r0[2];
            }
          }
          for (let a = 0; a < 6; a++) JTJ[a * 6 + a] *= 1.001;   // mild damping
          const delta = solveLinear(JTJ, JTr.map((v) => -v), 6);
          if (!delta) break;
          const [Rn, Cn] = apply(delta);
          R = Rn; C = Cn;
          if (Math.hypot(...delta) < 1e-9) break;
        }
      }
      return { R, C, inliers };
    }

    /** Angular inlier flags of a rig pose over union correspondences. */
    function rigInlierFlags(R, C, u, thAng) {
      const flags = new Uint8Array(u.obj.length);
      for (let k = 0; k < u.obj.length; k++) {
        const o = u.obj[k], b = u.brg[k];
        const dx = o[0] - C[0], dy = o[1] - C[1], dz = o[2] - C[2];
        const vx = R[0] * dx + R[1] * dy + R[2] * dz;
        const vy = R[3] * dx + R[4] * dy + R[5] * dz;
        const vz = R[6] * dx + R[7] * dy + R[8] * dz;
        const inv = 1 / (Math.hypot(vx, vy, vz) || 1e-12);
        if (Math.hypot(vx * inv - b[0], vy * inv - b[1], vz * inv - b[2]) < thAng * 1.5) flags[k] = 1;
      }
      return flags;
    }

    // per-rig registration diagnostics, returned on the recon for measurement
    const rigStats = [];

    /** Rig pose from the union of all faces' 2D-3D matches. The entry face's
     *  narrow-cone PnP is only a HYPOTHESIS: a wrong pose can fit matches in
     *  one 90-degree cone, but not bearings spread over the sphere. So
     *  spherical P3P RANSAC over the union proposes alternatives, the best
     *  start (by angular inliers) is refined jointly, and acceptance is gated
     *  on sphere-wide support BEFORE anything is committed to the map.
     *  Returns { R, C, inliers } (world->rig, centre) or null (rejected). */
    function solveRigPose(img, reg) {
      const rf = rigOf[img];
      const u = corrsForRig(rf.id);
      const R0 = tmul3(rf.R, reg.R);
      const C0 = centreOf(reg.R, reg.t);
      if (u.obj.length < 24) {
        // bootstrap: too few map points to judge sphere-wide support —
        // joint-refine if possible but keep the entry ungated (early rigs;
        // interim BA corrects)
        if (u.obj.length >= 12) {
          const jr0 = refineRigAngular(R0, C0, u.obj, u.brg, thN(img));
          if (jr0) return { R: jr0.R, C: jr0.C, inliers: jr0.inliers };
        }
        return { R: R0, C: C0, inliers: reg.inliers };
      }
      const thAng = thN(img);
      const count = (R, C) => {
        const fl = rigInlierFlags(R, C, u, thAng);
        let c2 = 0;
        for (let k = 0; k < fl.length; k++) c2 += fl[k];
        return c2;
      };
      let bestR = R0, bestC = C0, bestIn = count(R0, C0);
      const entryIn = bestIn;
      // spherical P3P RANSAC over the union
      const N = u.obj.length;
      let iters = 250;
      for (let it = 0; it < iters; it++) {
        const s0 = (rng() * N) | 0;
        let s1 = (rng() * N) | 0, s2 = (rng() * N) | 0;
        while (s1 === s0) s1 = (rng() * N) | 0;
        while (s2 === s0 || s2 === s1) s2 = (rng() * N) | 0;
        const cands = p3pBearings([u.obj[s0], u.obj[s1], u.obj[s2]], [u.brg[s0], u.brg[s1], u.brg[s2]]);
        for (const c2 of cands) {
          const inl = count(c2.R, c2.C);
          if (inl > bestIn) {
            bestIn = inl; bestR = c2.R; bestC = c2.C;
            const w = inl / N;
            const p = Math.max(1e-9, 1 - w * w * w);
            iters = Math.min(250, Math.max(it + 1, Math.ceil(Math.log(1e-3) / Math.log(p))));
          }
        }
      }
      const jr = refineRigAngular(bestR, bestC, u.obj, u.brg, thAng);
      if (!jr) return null;
      const flags = rigInlierFlags(jr.R, jr.C, u, thAng);
      let inl = 0;
      const inFaces = new Set(), unionFaces = new Set();
      for (let k = 0; k < flags.length; k++) {
        unionFaces.add(u.face[k]);
        if (flags[k]) { inl++; inFaces.add(u.face[k]); }
      }
      const st = {
        rig: rf.id, union: N, inliers: inl, frac: +(inl / N).toFixed(3),
        faces: inFaces.size, unionFaces: unionFaces.size, entryIn, ransacIn: bestIn,
      };
      rigStats.push(st);
      vlog(`rig ${rf.id}: joint pose over ${N} bearings (${inl} in, ${inFaces.size}/${unionFaces.size} faces, entry ${entryIn}, ransac ${bestIn})`);
      // NO acceptance gate here: measured on the bar360 walk, entry-time
      // inlier fractions of healthy frontier rigs (0.21-0.69) OVERLAP the
      // misregistered ones (0.28-0.61) — and rejecting a rig mid-growth
      // starves its neighbours (sequential walks died in cascades). Bad
      // rigs are caught by the sphere-wide audit AFTER the final BA, where
      // the separation is 40x (see the rig audit below).
      return { R: jr.R, C: jr.C, inliers: inl, frac: inl / N };
    }

    // ---- initialization pair: ranked by shared tracks ----
    // The distance weight (1 + |j - i|) favours wide pairs, which suppresses
    // capture-order neighbours entirely on a walk-around — yet those are the
    // natural seeds (the statue registered 30/34 from pair 29+30 and 23/34
    // from every wide pair, 2026-09-10). Keep the top wide pairs AND the top
    // adjacent pairs by shared count; the parallax-weighted score below
    // ranks them all the same way.
    const allPairs = [...sharedCount.entries()]
      .map(([k, c]) => ({ i: (k / 10000) | 0, j: k % 10000, c }))
      .filter((p) => p.c >= 30);
    const wide = allPairs.slice()
      .sort((a, b) => b.c * (1 + Math.abs(b.j - b.i)) - a.c * (1 + Math.abs(a.j - a.i)))
      .slice(0, 30);
    const near = allPairs.filter((p) => Math.abs(p.j - p.i) <= 2)
      .sort((a, b) => b.c - a.c)
      .slice(0, 10);
    const candPairs = [...wide, ...near.filter((p) => !wide.includes(p))]
      .filter((p) => !allowed || (allowed.has(p.i) && allowed.has(p.j)));

    const trackCorrs = (i, j) => {
      const x1s = [], x2s = [];
      for (const tr of tracks) {
        let oi = null, oj = null;
        for (const o of tr.obs) {
          if (!o.ok) continue;
          if (o.img === i) oi = o;
          if (o.img === j) oj = o;
        }
        if (oi && oj) { x1s.push(obsNorm(oi)); x2s.push(obsNorm(oj)); }
      }
      return { x1s, x2s };
    };

    // ---- GLOMAP-style global initialization ----
    // Solve all rotations at once (robust rotation averaging over pairwise
    // relative rotations), then all positions at once (translation-direction
    // registration). No incremental growth = no basin lock-in; the BA stack
    // downstream polishes. Opt-in: sfmOpts/opts.globalInit.
    if (useGlobal) {
      const t0g = performance.now();
      const gPairs = [...sharedCount.entries()]
        .map(([k, c]) => ({ i: (k / 10000) | 0, j: k % 10000, c }))
        .filter((p) => p.c >= 20);
      const rotEdges = [];
      const posRaw = [];
      for (const p of gPairs) {
        const { x1s, x2s } = trackCorrs(p.i, p.j);
        if (x1s.length < 20) continue;
        const favg = (K[p.i].f + K[p.j].f) / 2;
        const res = ransacE(x1s, x2s, 2.5 / favg, rng, 600);
        if (!res || res.inliers.length < 20 || res.inliers.length < 0.4 * x1s.length) continue;
        const pose = selectPose(decomposeE(res.U, res.V), x1s, x2s, res.inliers);
        if (!pose || pose.goodFrac < 0.6) continue;
        const w = Math.min(4, res.inliers.length / 50);
        rotEdges.push({ i: p.i, j: p.j, R: pose.R, w });
        // translation direction is meaningless without parallax
        if (pose.medianAngle > 0.01)
          posRaw.push({ i: p.i, j: p.j, t: pose.t, w: w * Math.min(1, pose.medianAngle / 0.03) });
      }
      vlog(`global init: ${rotEdges.length} rotation edges, ${posRaw.length} direction edges`);
      const ra = rotEdges.length >= n - 1 ? rotationAveraging(n, rotEdges, vlog) : null;
      if (!ra) { vlog('global init failed: rotation averaging'); return null; }
      // world direction of (C_i - C_j): the pair's t lives in cam-j coords,
      // so d_w = R_j^T t (using the AVERAGED R_j, not the noisy pairwise one)
      const posEdges = [];
      for (const e of posRaw) {
        if (!ra.R[e.i] || !ra.R[e.j]) continue;
        const d = m3mulv(m3t(ra.R[e.j]), e.t);
        const len = Math.hypot(d[0], d[1], d[2]);
        if (len < 1e-9) continue;
        posEdges.push({ i: e.i, j: e.j, d: [d[0] / len, d[1] / len, d[2] / len], w: e.w });
      }
      // camera-point rays: the tens of thousands of track observations are
      // what make positioning rigid on low-parallax video (camera-camera
      // directions alone gave 8-16% ATE; GLOMAP's core idea is exactly this
      // joint solve)
      const raysObs = [];
      let nPts = 0;
      for (const tr of tracks) {
        const ok = tr.obs.filter((o) => o.ok && ra.R[o.img]);
        if (ok.length < 2) continue;
        const k = nPts++;
        for (const o of ok) {
          const xn = feats[o.img].xn[o.feat], yn = feats[o.img].yn[o.feat];
          const nr = Math.hypot(xn, yn, 1);
          const vc = [xn / nr, yn / nr, 1 / nr];
          const R = ra.R[o.img]; // world ray = R^T vc
          raysObs.push({
            i: o.img, k,
            v: [
              R[0] * vc[0] + R[3] * vc[1] + R[6] * vc[2],
              R[1] * vc[0] + R[4] * vc[1] + R[7] * vc[2],
              R[2] * vc[0] + R[5] * vc[1] + R[8] * vc[2],
            ],
          });
        }
      }
      const jp = (posEdges.length >= 3 && raysObs.length >= 100)
        ? globalPositionsJoint(n, posEdges, raysObs, nPts, ra.anchor, vlog)
        : null;
      if (!jp || !jp.C) { vlog('global init failed: positioning'); return null; }
      const Cs = jp.C;
      let reg = 0;
      for (let i = 0; i < n; i++) {
        if (!ra.R[i] || !Cs[i]) continue;
        const R = ra.R[i], C = Cs[i];
        poses[i] = {
          R: Array.from(R),
          t: [
            -(R[0] * C[0] + R[1] * C[1] + R[2] * C[2]),
            -(R[3] * C[0] + R[4] * C[1] + R[5] * C[2]),
            -(R[6] * C[0] + R[7] * C[1] + R[8] * C[2]),
          ],
        };
        registered.add(i);
        reg++;
      }
      vlog(`global init: ${reg}/${n} cameras in ${((performance.now() - t0g) / 1000).toFixed(1)}s`);
      if (reg < 3) return null;
      for (const tr of tracks) triangulateTrack(tr);
      globalRefine(2);
      await tick();
    }

    const scored = [];
    for (const p of useGlobal ? [] : candPairs) {
      // same-rig faces share a centre: zero baseline, no triangulation —
      // never a seed (the parallax gate below also rejects them, this just
      // saves the RANSAC)
      if (rigOf && rigOf[p.i] && rigOf[p.j] && rigOf[p.i].id === rigOf[p.j].id) continue;
      const { x1s, x2s } = trackCorrs(p.i, p.j);
      if (x1s.length < 30) continue;
      const favg = (K[p.i].f + K[p.j].f) / 2;
      const res = ransacE(x1s, x2s, 2.5 / favg, rng);
      if (!res || res.inliers.length < 30) continue;
      const pose = selectPose(decomposeE(res.U, res.V), x1s, x2s, res.inliers);
      if (!pose || pose.goodFrac < 0.7 || pose.medianAngle < 0.015) continue;
      // cap the parallax bonus so a sparse wide-baseline pair can't beat a
      // dense moderate one. The cap was 0.06 rad (3.4 deg) — MEASURED
      // 2026-09-09 on truck-251: a 3.7-deg pair with 1126 inliers then
      // outscored an 11.7-deg pair with 1028, the chain started bent
      // (rms 7.4 -> 4.1 px at 12 cams instead of 5.4 -> 1.8) and finished at
      // ATE 1.4 % / 21.5 dB instead of 0.00 % / 25.9. Parallax keeps counting
      // up to 0.2 rad (11.5 deg); a 200-inlier 20-deg pair still loses to a
      // 1000-inlier 8-deg one.
      scored.push({
        p, pose, nInl: res.inliers.length,
        score: res.inliers.length * Math.min(pose.medianAngle, 0.2),
      });
    }
    // A near-degenerate init pair poisons EVERYTHING downstream: camping once
    // initialized from a 1.1-deg-parallax pair and the whole reconstruction
    // collapsed (path length 0.36 vs 44) while every internal check still
    // passed (rms 0.67px!). Prefer >= 5 deg median parallax when such
    // candidates exist, then >= 1.4 deg; the low-parallax tier is only a
    // last resort.
    {
      const wide = scored.filter((s) => s.pose.medianAngle >= 0.087);
      const strong = wide.length ? wide : scored.filter((s) => s.pose.medianAngle >= 0.025);
      if (strong.length) { scored.length = 0; scored.push(...strong); }
    }
    scored.sort((a, b) => b.score - a.score);
    // opts.initPair = [i, j]: seed from a named pair when it is a candidate
    // (dev / bench tool — reproduces a registration that a lucky ranking gave)
    if (Array.isArray(sfmOpts.initPair)) {
      const [pi0, pj0] = sfmOpts.initPair;
      const k = scored.findIndex((s) => (s.p.i === pi0 && s.p.j === pj0) || (s.p.i === pj0 && s.p.j === pi0));
      if (k > 0) scored.unshift(...scored.splice(k, 1));
      vlog(`init pair request ${pi0}+${pj0}: ${k >= 0 ? 'ranked first' : 'not a candidate'}`);
    }
    if (!useGlobal && !scored.length) return null;

    let initDone = false;
    // initSkip: start from the k-th ranked pair (init trials, see initTrials)
    for (const { p, pose, nInl } of scored.slice(Math.min(initSkip, Math.max(0, scored.length - 1)))) {
      poses[p.i] = { R: I3(), t: [0, 0, 0] };
      poses[p.j] = { R: pose.R, t: pose.t.slice() };
      registered.add(p.i); registered.add(p.j);
      let tri = 0;
      for (const tr of tracks) {
        const hasI = tr.obs.some((o) => o.ok && o.img === p.i);
        const hasJ = tr.obs.some((o) => o.ok && o.img === p.j);
        if (hasI && hasJ && triangulateTrack(tr)) tri++;
      }
      if (tri < 30) {
        poses[p.i] = poses[p.j] = null;
        registered.clear();
        for (const tr of tracks) { tr.X = null; for (const o of tr.obs) o.ok = true; }
        continue;
      }
      vlog(`init pair: images ${p.i} + ${p.j} (${nInl} E-inliers, ${tri} points, ` +
           `median parallax ${(pose.medianAngle * 180 / Math.PI).toFixed(1)} deg)`);
      // the init pair is registered too — the overlay's ring was two frustums short without this
      for (const im of [p.i, p.j]) ev({ stage: 'register', done: registered.size, total: n, detail: { image: im, R: Array.from(poses[im].R), t: Array.from(poses[im].t), f: K[im].f, cloud: null, points: tri, final: !!withBA } });
      initDone = true;
      break;
    }
    if (!useGlobal && !initDone) return null;

    // ---- incremental registration ----
    function corrsFor(img) {
      const obj = [], im = [];
      const ft = featTrack[img];
      for (let f = 0; f < feats[img].n; f++) {
        const tid = ft[f];
        if (tid < 0) continue;
        const tr = tracks[tid];
        if (!tr.X) continue;
        const o = tr.obs.find((oo) => oo.img === img);
        if (!o || !o.ok) continue;
        obj.push(tr.X);
        im.push([feats[img].xn[f], feats[img].yn[f]]);
      }
      return { obj, im };
    }

    function refineCamera(img) {
      const { obj, im } = corrsFor(img);
      if (obj.length < 6) return;
      poses[img] = refinePose(poses[img].R, poses[img].t, obj, im, 8);
    }

    function globalRefine(rounds) {
      for (let r = 0; r < rounds; r++) {
        for (const img of registered) refineCamera(img);
        for (const tr of tracks) {
          if (tr.X) {
            for (const o of tr.obs) {
              if (!o.ok || !poses[o.img]) continue;
              const err = reprojError(poses[o.img].R, poses[o.img].t, tr.X, obsNorm(o));
              if (!(err < thN(o.img) * 1.5)) o.ok = false;
            }
          }
          triangulateTrack(tr);
        }
      }
    }

    /** K changed (BA rescales f) or feats.x/y moved (LK refine): the cached
     *  normalized coords everything else uses must follow. */
    function refreshNorm() {
      for (let i = 0; i < n; i++) {
        const f = feats[i];
        for (let k = 0; k < f.n; k++) {
          f.xn[k] = (f.x[k] - K[i].cx) / K[i].f;
          f.yn[k] = (f.y[k] - K[i].cy) / K[i].f;
        }
      }
    }

    /** Joint sparse BA over the whole current reconstruction. Applies the
     *  result (poses, shared f, point positions) and returns the ba result,
     *  or null when the problem is too small / dimensions are mixed. */
    async function runGlobalBA(tag, o = {}) {
      if (registered.size < 8) return null;
      const regList = [...registered].sort((a, b) => a - b);
      const sameDims = regList.every((img) =>
        images[img].fw === images[regList[0]].fw && images[img].fh === images[regList[0]].fh);
      if (!sameDims) { vlog(`BA ${tag} skipped: mixed image dimensions`); return null; }
      const camMap = new Map(regList.map((img, i) => [img, i]));
      const baCams = regList.map((img) => ({ R: Array.from(poses[img].R), t: poses[img].t.slice() }));
      const baPoints = [], baTracks = [], baObs = [];
      // o.maxPoints: an interim BA only needs enough points to hold the
      // chain straight — every stride-th triangulated track (deterministic),
      // so a 96 k-point model adjusts on ~25 k. The final BAs use everything.
      let stride = 1;
      if (o.maxPoints > 0) {
        let nTri = 0;
        for (const tr of tracks) if (tr.X) nTri++;
        if (nTri > o.maxPoints) stride = Math.ceil(nTri / o.maxPoints);
      }
      let ti = 0;
      for (const tr of tracks) {
        if (!tr.X) continue;
        if (stride > 1 && (ti++ % stride) !== 0) continue;
        const pi = baPoints.length;
        let nOk = 0;
        for (const ob of tr.obs) {
          if (!ob.ok || !poses[ob.img]) continue;
          baObs.push({ ci: camMap.get(ob.img), pi, u: feats[ob.img].x[ob.feat], v: feats[ob.img].y[ob.feat] });
          nOk++;
        }
        if (nOk < 2) { baObs.length -= nOk; continue; }
        baPoints.push(tr.X.slice());
        baTracks.push(tr);
      }
      if (baPoints.length < 50) return null;
      const t0 = performance.now();
      // rigs enter BA as ONE 6-DOF block per rig — no per-face pose exists
      const baRig = rigOf ? regList.map((img) => rigOf[img] ? { id: rigOf[img].id, Rf: rigOf[img].R } : null) : null;
      const res = await bundleAdjust(
        { cams: baCams, points: baPoints, obs: baObs, camRig: baRig,
          f: K[regList[0]].f, cx: K[regList[0]].cx, cy: K[regList[0]].cy },
        { maxIters: o.maxIters ?? 30, huberPx: 1.5 * pxScaleOf(images[0]),
          // lockIntrinsics (2026-09-06, the rig default via session.solve): sliced cubemap
          // faces know f, k1 = k2 = 0 and a square pixel EXACTLY by construction — refining
          // them only fits noise (bar360 30k: +0.29 dB locked). bench ?lockk=1 forces it on.
          refineF: !sfmOpts.lockIntrinsics,
          refineDistortion: sfmOpts.lockIntrinsics ? false : (o.refineDistortion ?? (sfmOpts.refineDistortion ?? true)),
          refineAspect: sfmOpts.lockIntrinsics ? false : (sfmOpts.refineAspect ?? false),
          // phones: yield the main thread between LM iterations (uiYield);
          // desktop stays a straight synchronous burn
          yieldFn: sfmOpts.uiYield ? tick : undefined,
          log: o.verbose ? vlog : () => {} });
      vlog(`BA ${tag} (${baCams.length} cams, ${baPoints.length} pts, ${baObs.length} obs) ` +
           `in ${((performance.now() - t0) / 1000).toFixed(1)}s: ` +
           `rms ${res.rmsBefore.toFixed(2)} -> ${res.rmsAfter.toFixed(2)}px, ` +
           `f x${res.fScale.toFixed(4)}, k1 ${res.k1.toFixed(4)}, k2 ${res.k2.toFixed(4)}` +
           (res.aspect !== 1 ? `, aspect ${res.aspect.toFixed(4)}` : ''));
      regList.forEach((img, i) => { poses[img] = { R: baCams[i].R, t: baCams[i].t }; });
      if (Math.abs(res.fScale - 1) > 1e-9) {
        for (let i = 0; i < n; i++) K[i].f *= res.fScale;
        refreshNorm();
      }
      baTracks.forEach((tr, j) => { tr.X = baPoints[j]; });
      return res;
    }

    /** Iterative outlier rejection (what COLMAP does between BA rounds).
     *  Measured on truck vs COLMAP ground truth: ~25% of our observations are
     *  outliers (p90 17px under the true geometry). Huber only softens them —
     *  left in the problem they bend the whole trajectory (7% ATE); the same
     *  BA holds the true solution at 0.3% ATE once they are filtered. */
    function baFilterObs(k1v, k2v, tag) {
      const items = [];
      const errsAll = [];
      for (const tr of tracks) {
        if (!tr.X) continue;
        for (const o of tr.obs) {
          if (!o.ok || !poses[o.img]) continue;
          const R = poses[o.img].R, t = poses[o.img].t, X = tr.X;
          const pz = R[6]*X[0] + R[7]*X[1] + R[8]*X[2] + t[2];
          let e = 1e9;
          if (pz > 1e-9) {
            const x = (R[0]*X[0] + R[1]*X[1] + R[2]*X[2] + t[0]) / pz;
            const y = (R[3]*X[0] + R[4]*X[1] + R[5]*X[2] + t[1]) / pz;
            const r2 = x*x + y*y;
            const D = 1 + k1v*r2 + k2v*r2*r2;
            const du = feats[o.img].x[o.feat] - (K[o.img].f*x*D + K[o.img].cx);
            const dv = feats[o.img].y[o.feat] - (K[o.img].f*y*D + K[o.img].cy);
            e = Math.hypot(du, dv);
          }
          items.push({ o, e });
          errsAll.push(e);
        }
      }
      if (!items.length) return 0;
      errsAll.sort((a, b) => a - b);
      const med = errsAll[errsAll.length >> 1];
      const TH = Math.max(2.0, 3 * med);
      let dropped = 0;
      for (const it of items) if (it.e > TH) { it.o.ok = false; dropped++; }
      vlog(`  obs filter (${tag}): dropped ${dropped}/${items.length} obs > ${TH.toFixed(1)}px (med ${med.toFixed(2)})`);
      return dropped / items.length;
    }

    /** Periodic global BA during incremental registration. Coordinate-descent
     *  refinement alone lets drift accumulate until the reconstruction settles
     *  into a bent basin that no final BA can escape (two self-consistent
     *  minima at ~0.6px rms exist on the truck set; only one matches COLMAP).
     *  Frequent joint solves + filtering keep growth inside the right basin. */
    async function interimBA() {
      // interimBAMaxPoints (default 25000): subsample the tracks entering an
      // interim BA — with the upsampled-octave features truck-251 carries
      // 96 k points / 514 k observations and 40 interim BAs cost 270 s of a
      // 430 s final pass; 0 = all points (the pre-2026-09-09 behaviour)
      const res = await runGlobalBA(`interim @${registered.size}`,
        // MEASURED 2026-09-09: all points 268 s / 25.88 dB, 25 k 129 s / 25.93,
        // 10 k 78 s / 26.03 — the chain stays straight on ~40 points per camera
        { maxIters: 12, refineDistortion: false, maxPoints: sfmOpts.interimBAMaxPoints ?? 10000 });
      if (!res) return;
      for (const tr of tracks) triangulateTrack(tr);
      baFilterObs(res.k1, res.k2, `interim @${registered.size}`);
      for (const tr of tracks) triangulateTrack(tr);
    }

    /** Register from the pose of the registered image sharing the most
     *  tracks, then alternate LM refinement and inlier re-selection.
     *  Falls back to DLT-PnP RANSAC. */
    function registerImage(img) {
      const { obj, im } = corrsFor(img);
      if (obj.length < 8) return null;
      const th = thN(img) * 1.5;

      const shared = new Map();
      const ft = featTrack[img];
      for (let f = 0; f < feats[img].n; f++) {
        const tid = ft[f];
        if (tid < 0 || !tracks[tid].X) continue;
        for (const o of tracks[tid].obs) {
          if (o.ok && o.img !== img && poses[o.img])
            shared.set(o.img, (shared.get(o.img) || 0) + 1);
        }
      }
      let ref = -1, refCount = 0;
      for (const [r, c] of shared) if (c > refCount) { refCount = c; ref = r; }

      const tryFrom = (R0, t0) => {
        let R = R0, t = t0;
        let curObj = obj, curIm = im;
        let inl = 0;
        for (let round = 0; round < 4; round++) {
          const p = refinePose(R, t, curObj, curIm, 10);
          R = p.R; t = p.t;
          const io = [], ii = [];
          for (let k = 0; k < obj.length; k++)
            if (reprojError(R, t, obj[k], im[k]) < th) { io.push(obj[k]); ii.push(im[k]); }
          inl = io.length;
          if (inl < 8) return null;
          curObj = io; curIm = ii;
        }
        return { R, t, inliers: inl };
      };

      if (ref >= 0 && refCount >= 8) {
        const res = tryFrom(poses[ref].R, poses[ref].t.slice());
        if (res && res.inliers >= Math.max(8, obj.length * 0.25)) return res;
      }
      const pnp = pnpRansac(obj, im, th, rng);
      if (pnp) return { R: pnp.R, t: pnp.t, inliers: pnp.inliers.length };
      return null;
    }

    // Registration in passes: failed images get another chance as the
    // reconstruction grows.
    for (let pass = 0; pass < 5; pass++) {
      failed.clear();
      let addedThisPass = 0;
      let sinceRefine = 0;
      let sinceBA = 0;
      let lastBA = 0;   // camera count at the last interim BA (geometric cadence)
      // A frame that fails registration is not lost for good: it failed against
      // the model of THAT moment (few triangulated tracks, an early pose guess).
      // When no candidate is left, frames whose triangulated support has grown
      // 1.5x since they failed get another attempt (up to 3 rounds). Long forward
      // walks (charleston: 114/249 registered, 306/502 on the dense set) stall
      // exactly this way — the chain breaks once and everything beyond it is
      // tried once, too early, then abandoned. 2026-09-08.
      const failedAt = new Map();
      let retryRounds = 0;
      while (registered.size + failed.size < n) {
        let bestImg = -1, bestCount = 0;
        for (let img = 0; img < n; img++) {
          if (registered.has(img) || failed.has(img)) continue;
          if (allowed && !allowed.has(img)) continue;
          let c = 0;
          const ft = featTrack[img];
          for (let f = 0; f < feats[img].n; f++) {
            const tid = ft[f];
            if (tid >= 0 && tracks[tid].X) c++;
          }
          if (c > bestCount) { bestCount = c; bestImg = img; }
        }
        if (bestImg < 0 || bestCount < 8) {
          let revived = 0;
          if (retryRounds < 3 && failed.size) {
            for (const img of [...failed]) {
              let c = 0;
              const ft = featTrack[img];
              for (let f = 0; f < feats[img].n; f++) { const tid = ft[f]; if (tid >= 0 && tracks[tid].X) c++; }
              if (c >= 8 && c >= 1.5 * (failedAt.get(img) || 0)) { failed.delete(img); revived++; }
            }
            retryRounds++;
          }
          if (!revived) break;
          vlog(`retrying ${revived} failed frame(s) whose triangulated support grew (round ${retryRounds})`);
          continue;
        }

        const reg = registerImage(bestImg);
        if (!reg) { failed.add(bestImg); failedAt.set(bestImg, bestCount); continue; }
        // rig entry: solve the RIG pose sphere-wide BEFORE committing —
        // a wrong narrow-cone pose must not triangulate junk points or
        // drag five siblings along with it (see solveRigPose)
        let rigPose = null;
        if (rigOf && rigOf[bestImg]) {
          rigPose = solveRigPose(bestImg, reg);
          if (!rigPose) { failed.add(bestImg); failedAt.set(bestImg, bestCount); continue; }
          const Rb = mul3(rigOf[bestImg].R, rigPose.R);
          const C = rigPose.C;
          reg.R = Rb;
          reg.t = [
            -(Rb[0] * C[0] + Rb[1] * C[1] + Rb[2] * C[2]),
            -(Rb[3] * C[0] + Rb[4] * C[1] + Rb[5] * C[2]),
            -(Rb[6] * C[0] + Rb[7] * C[1] + Rb[8] * C[2]),
          ];
          reg.inliers = rigPose.inliers;
        }
        poses[bestImg] = { R: reg.R, t: reg.t };
        registered.add(bestImg);
        addedThisPass++;

        let newPts = 0;
        const ft = featTrack[bestImg];
        for (let f = 0; f < feats[bestImg].n; f++) {
          const tid = ft[f];
          if (tid < 0 || tracks[tid].X) continue;
          if (triangulateTrack(tracks[tid])) newPts++;
        }
        vlog(`registered image ${bestImg} (${reg.inliers}/${bestCount} inliers, +${newPts} points)`);
        if (withBA) {
          // a downsampled snapshot of the triangulated cloud rides along so a
          // UI can show the reconstruction growing, not just the frustums —
          // with real colours (sampleColor is a plain array read, ~free)
          const cloud = [];
          const cloudRgb = [];
          let nPts = 0;
          const stride = Math.max(1, (tracks.length / 4000) | 0);
          for (let ti = 0; ti < tracks.length; ti++) {
            const X = tracks[ti].X;
            if (!X) continue;
            if (nPts % stride === 0) {
              cloud.push(X[0], X[1], X[2]);
              const o = tracks[ti].obs.find((ob) => ob.ok) || tracks[ti].obs[0];
              const c = sampleColor(o.img, feats[o.img].x[o.feat], feats[o.img].y[o.feat]);
              cloudRgb.push(c[0], c[1], c[2]);
            }
            nPts++;
          }
          ev({ stage: 'register', done: registered.size, total: n, detail: {
            image: bestImg, R: Array.from(reg.R), t: Array.from(reg.t), f: K[bestImg].f,
            cloud, cloudRgb, points: nPts, final: !!withBA,
          } });
        }

        // rig propagation: siblings share the centre and sit at fixed, known
        // rotations — pose them all from the sphere-wide rig pose (already
        // hypothesized, refined, and gated in solveRigPose before commit).
        // Sibling observations then join the map and triangulate against
        // OTHER rigs (the parallax gate rejects the zero-baseline same-rig
        // combinations).
        if (rigPose) {
          const rf = rigOf[bestImg];
          const Rr = rigPose.R;                     // world->rig
          const C = rigPose.C;
          for (let j = 0; j < n; j++) {
            if (j === bestImg || registered.has(j)) continue;
            if (!rigOf[j] || rigOf[j].id !== rf.id) continue;
            const Rj = mul3(rigOf[j].R, Rr);
            const tj = [
              -(Rj[0] * C[0] + Rj[1] * C[1] + Rj[2] * C[2]),
              -(Rj[3] * C[0] + Rj[4] * C[1] + Rj[5] * C[2]),
              -(Rj[6] * C[0] + Rj[7] * C[1] + Rj[8] * C[2]),
            ];
            poses[j] = { R: Rj, t: tj };
            registered.add(j);
            failed.delete(j);      // a face that failed PnP is still rig-posed
            addedThisPass++;
            let np = 0;
            const ftj = featTrack[j];
            for (let f2 = 0; f2 < feats[j].n; f2++) {
              const tid = ftj[f2];
              if (tid < 0 || tracks[tid].X) continue;
              if (triangulateTrack(tracks[tid])) np++;
            }
            vlog(`rig ${rf.id}: pose propagated to image ${j} (+${np} points)`);
            if (withBA) {
              ev({ stage: 'register', done: registered.size, total: n, detail: {
                image: j, R: Array.from(Rj), t: tj.slice(), f: K[j].f,
              } });
            }
          }
        }

        if (++sinceRefine >= 3) { globalRefine(1); sinceRefine = 0; }
        // periodic joint BA so the growing chain never drifts into a bent
        // basin (only on the final run — the focal search stays cheap).
        // Cadence: every 6 registrations (default). MEASURED 2026-09-09: a
        // geometric cadence (interimBARatio 1.15, COLMAP-style) cut the 40
        // interim BAs to 19 and the precise truck solve 10.8 -> 6.9 min, but
        // the chain dived between the sparser adjustments (rms 22.7 px before
        // the last one, ATE 1.77 % vs 0.00 %, 21.9 dB at 30k vs 25.9). The
        // every-6 cadence stays; the interim BAs get cheaper instead
        // (interimBAMaxPoints, see interimBA). interimBARatio > 0 opts in.
        ++sinceBA;
        const baRatio = sfmOpts.interimBARatio ?? 0;
        const baDue = registered.size >= 12 && (baRatio > 0
          ? registered.size >= Math.max(lastBA + 6, Math.ceil(lastBA * baRatio))
          : sinceBA >= 6);
        if (withBA && sfmOpts.interimBA !== false && baDue) {
          interimBA();
          sinceRefine = 0;
          sinceBA = 0;
          lastBA = registered.size;
        }
        await tick();
        checkAbort();
      }
      if (registered.size === n) break;
      globalRefine(1);
      if (addedThisPass === 0) break;
      await tick();
    }
    globalRefine(2);
    await tick();

    // ---- true sparse bundle adjustment (final pass only) ----
    // The alternating refine above is coordinate descent and cannot remove
    // smooth global trajectory bends; the joint LM solve (with shared focal
    // and radial distortion) can — this is what makes COLMAP paths clean.
    let baResult = null;
    if (withBA && sfmOpts.ba !== false && registered.size >= 3) {
      ev({ stage: 'ba', done: 0, total: 1 });
      if (sfmOpts.lkRefine !== false) { refineObsLK(images, feats, tracks, poses, vlog); refreshNorm(); }
      baResult = await runGlobalBA('pass 1', { verbose: true });
      if (baResult && sfmOpts.obsFilter !== false) {
        if (baFilterObs(baResult.k1, baResult.k2, 'after pass 1') > 0.002)
          baResult = (await runGlobalBA('pass 1b', { verbose: true })) || baResult;
      }
      // ---- guided track extension ----
      // Mean track length ~3 obs makes the camera chain bendable; project
      // each adjusted point into every registered frame it is not yet
      // observed in and claim descriptor-verified features nearby. Requires
      // pass-1 distortion so edge-of-frame projections land where the raw
      // (distorted) features actually are.
      if (baResult && sfmOpts.trackExtend !== false) {
        const regList = [...registered].sort((a, b) => a - b);
        // two rounds: the second round extends from the improved pass-2 poses
        // and reaches frames the first projection missed
        for (let round = 1; round <= 2; round++) {
          const extObs = extendTracks(feats, tracks, poses, K, regList,
                                      baResult.k1, baResult.k2, images[regList[0]], vlog);
          if (!extObs.length) break;
          // refine the freshly added obs to subpixel too (idempotent for
          // already-refined ones — they are at their converged position)
          if (sfmOpts.lkRefine !== false) { refineObsLK(images, feats, tracks, poses, vlog); refreshNorm(); }
          baResult = (await runGlobalBA(`pass ${1 + round}`, { verbose: true })) || baResult;
        }
      }
      if (baResult && sfmOpts.obsFilter !== false) {
        // iterate filter -> BA until the drop rate is negligible (each round
        // reveals outliers the previous bent model was hiding)
        for (let round = 1; round <= 3; round++) {
          if (baFilterObs(baResult.k1, baResult.k2, `final round ${round}`) <= 0.002) break;
          baResult = (await runGlobalBA(`final ${round}`, { verbose: true })) || baResult;
        }
      }
      await tick();
    }

    // ---- sphere-wide rig audit (final map only) ----
    // A misregistered rig is self-consistent on its own island of points but
    // reprojects the SHARED map badly. Measured on the bar360 walk (76 rigs):
    // healthy rigs fit >= 95% of their observations of triangulated tracks
    // (population median 0.97, median residual 0.22x threshold); the
    // misregistered ones fit 3-42% at 8-99x threshold — a 40x separation
    // that entry-time stats never show. Flag rigs under 0.6x the population
    // median, re-register them against the mature map (spherical P3P RANSAC
    // + joint refine over the union of all faces), and drop them if the
    // retry still fails the bar: a wrong rig poisons training harder than a
    // missing one. Needs enough rigs for the median to mean something.
    const auditRigs = () => {
      const audit = new Map();
      for (const tr of tracks) {
        if (!tr.X) continue;
        for (const o of tr.obs) {
          if (!rigOf[o.img] || !poses[o.img]) continue;
          const rid = rigOf[o.img].id;
          let a = audit.get(rid);
          if (!a) audit.set(rid, a = { nObs: 0, nFit: 0 });
          a.nObs++;
          if (reprojError(poses[o.img].R, poses[o.img].t, tr.X, obsNorm(o)) < thN(o.img) * 1.5) a.nFit++;
        }
      }
      return audit;
    };
    if (withBA && rigOf && baResult && sfmOpts.rigAudit !== false) {
      const membersOf = (rid) => {
        const m = [];
        for (let i = 0; i < n; i++) if (rigOf[i] && rigOf[i].id === rid) m.push(i);
        return m;
      };
      for (let round = 1; round <= 2; round++) {
        const audit = auditRigs();
        if (audit.size < 8) break;
        const fracs = [...audit.values()].map((a) => a.nFit / a.nObs).sort((x, y) => x - y);
        const medFit = fracs[fracs.length >> 1];
        const flagged = [];
        for (const [rid, a] of audit)
          if (a.nFit / a.nObs < medFit * 0.6) flagged.push(rid);
        if (!flagged.length) break;
        vlog(`rig audit: ${flagged.length} rig(s) under ${(medFit * 0.6).toFixed(2)} fit (median ${medFit.toFixed(2)}): ${flagged.join(', ')}`);
        // de-register ALL flagged rigs first so none anchors another's retry
        for (const rid of flagged) {
          for (const i of membersOf(rid)) { poses[i] = null; registered.delete(i); }
        }
        const touched = [];
        for (const tr of tracks) {
          let hit = false;
          for (const o of tr.obs)
            if (rigOf[o.img] && flagged.includes(rigOf[o.img].id)) { o.ok = false; hit = true; }
          if (hit) touched.push(tr);
        }
        for (const tr of touched) if (tr.X) triangulateTrack(tr);
        // observations become hypotheses again for the re-registration
        // (unregistered faces contribute nothing until posed; the post-
        // rescue BA filter re-drops the liars)
        for (const tr of touched)
          for (const o of tr.obs)
            if (rigOf[o.img] && flagged.includes(rigOf[o.img].id)) o.ok = true;
        // retry each flagged rig against the mature map
        let rescued = 0;
        for (const rid of flagged) {
          let ok = false;
          for (const i of membersOf(rid)) {
            const reg = registerImage(i);
            if (!reg) continue;
            const rp = solveRigPose(i, reg);
            if (!rp || !(rp.frac >= medFit * 0.6)) continue;
            for (const j of membersOf(rid)) {
              const Rj = mul3(rigOf[j].R, rp.R);
              poses[j] = { R: Rj, t: [
                -(Rj[0] * rp.C[0] + Rj[1] * rp.C[1] + Rj[2] * rp.C[2]),
                -(Rj[3] * rp.C[0] + Rj[4] * rp.C[1] + Rj[5] * rp.C[2]),
                -(Rj[6] * rp.C[0] + Rj[7] * rp.C[1] + Rj[8] * rp.C[2]),
              ] };
              registered.add(j);
            }
            ok = true;
            rescued++;
            vlog(`rig ${rid}: re-registered by audit (fit ${rp.frac.toFixed(2)})`);
            break;
          }
          if (!ok) {
            // stays out — kill its obs for good so triangulations are clean
            for (const i of membersOf(rid)) {
              poses[i] = null; registered.delete(i);
              const ft = featTrack[i];
              for (let f2 = 0; f2 < feats[i].n; f2++) {
                const tid = ft[f2];
                if (tid < 0) continue;
                for (const o of tracks[tid].obs) if (o.img === i) o.ok = false;
              }
            }
            vlog(`rig ${rid}: DROPPED by audit (re-registration failed)`);
          }
        }
        for (const tr of touched) triangulateTrack(tr);
        baResult = (await runGlobalBA(`rig audit ${round}`, { verbose: true })) || baResult;
        if (sfmOpts.obsFilter !== false) baFilterObs(baResult.k1, baResult.k2, `rig audit ${round}`);
        for (const tr of tracks) if (tr.X) triangulateTrack(tr);
        if (!rescued) break;   // nothing re-entered; a second round can't improve
      }
      await tick();
    }

    // ---- collect output + median reprojection error ----
    const points = [];
    const errs = [];
    for (const tr of tracks) {
      if (!tr.X) continue;
      const okObs = tr.obs.filter((o) => o.ok && poses[o.img]);
      if (okObs.length < 2) continue;
      let r = 0, g = 0, b = 0, c = 0;
      for (const o of okObs.slice(0, 3)) {
        const rgb = sampleColor(o.img, feats[o.img].x[o.feat], feats[o.img].y[o.feat]);
        r += rgb[0]; g += rgb[1]; b += rgb[2]; c++;
        errs.push(reprojError(poses[o.img].R, poses[o.img].t, tr.X, obsNorm(o)) * K[o.img].f);
      }
      points.push({ X: tr.X, rgb: [r / c, g / c, b / c], nObs: okObs.length });
    }
    if (opts.denseSeed) {
      // Seed-only densification: tracks the strict pass rejected (low
      // parallax, pruned observations) get one relaxed retry — 3x the
      // reprojection budget, parallax floor 0.0015 rad, 2 obs suffice.
      // Poses are FROZEN here, so a loose point can only mis-seed a splat
      // the optimizer will move, never bend geometry. COLMAP triangulates
      // ~5x more points than our strict pass on the same scene; the seed
      // cloud (not the poses) was measured worth ~+0.15 dB on truck.
      let added = 0;
      for (const tr of tracks) {
        if (tr.X) continue;
        const cams2 = [], obsPts = [], used = [];
        for (const o of tr.obs) {
          if (!poses[o.img]) continue; // o.ok ignored: strict-pruned obs may pass looser bounds
          cams2.push(poses[o.img]); obsPts.push(obsNorm(o)); used.push(o);
        }
        if (cams2.length < 2) continue;
        const X = triangulateN(cams2, obsPts);
        if (!X) continue;
        let ok = 0;
        for (let k = 0; k < cams2.length; k++) {
          if (reprojError(cams2[k].R, cams2[k].t, X, obsPts[k]) < 3 * thN(used[k].img)) ok++;
        }
        if (ok < 2) continue;
        let maxAng = 0;
        for (let a = 0; a < cams2.length - 1 && maxAng < 0.0015; a++)
          for (let b2 = a + 1; b2 < cams2.length && maxAng < 0.0015; b2++)
            maxAng = Math.max(maxAng, parallaxAngle(cams2[a], cams2[b2], X));
        if (maxAng < 0.0015) continue;
        const o0 = used[0];
        points.push({ X, rgb: sampleColor(o0.img, feats[o0.img].x[o0.feat], feats[o0.img].y[o0.feat]), nObs: ok });
        added++;
      }
      log(`  seed densification: +${added} relaxed points (${points.length} total)`);
    }
    errs.sort((a, b) => a - b);
    const medErr = errs.length ? errs[errs.length >> 1] : Infinity;

    const cams = [];
    for (const img of registered) {
      cams.push({
        imgIdx: img,
        R: poses[img].R, t: poses[img].t,
        f: K[img].f, cx: K[img].cx, cy: K[img].cy,
        // non-square pixels (refineAspect): fy = f · aspect, consumed by the trainer's per-axis focal
        ...(baResult && baResult.aspect && baResult.aspect !== 1 ? { fy: K[img].f * baResult.aspect } : {}),
      });
    }
    cams.sort((a, b) => a.imgIdx - b.imgIdx);

    // internals hook for dev tooling / UI beats (feature marks, match lines)
    if (opts.debug) opts.debug({ feats, tracks, poses, K, registered, corrsFor, thN });

    return {
      cams, points, medErr, fScale,
      k1: baResult ? baResult.k1 : 0,
      k2: baResult ? baResult.k2 : 0,
      aspect: baResult ? (baResult.aspect ?? 1) : 1,
      fFeat: K[0].f,
      rmsBA: baResult ? baResult.rmsAfter : null,
      rigStats: rigOf ? rigStats : undefined,
    };
  }

  // ---- focal search ----
  // Evaluate ALL candidates (an early-exit at the first full registration
  // once picked a focal 24% off on the synthetic GT while a near-perfect
  // candidate sat later in the grid). Rank by registered cameras, then by
  // reprojection error. Incremental runs rank the tiebreak by ANGULAR error
  // (raw pixel error interacts with the 1/f registration thresholds). Global
  // init registers everything at every focal, so the count never
  // discriminates and angular error trivially favors the largest f (it
  // divides by f) — measured: it picked 1.56x over the true 0.78x and BA
  // invented k1=0.70 to cope. There the PIXEL median is the right metric
  // (detector noise is in pixels; model misfit adds to it).
  const useGlobalSearch = !!opts.globalInit;

  // Init trials (2026-09-10): on a small set the whole reconstruction hangs
  // on the first init pair — the 34-photo statue registered 30/34 from pair
  // 29+30 and 18/34 from pair 2+3 with identical everything else. For sets
  // up to 60 images the top-ranked pairs are each tried with the cheap
  // registration (no BA, a second or two each) and the one that places the
  // most cameras (then the lower pixel median) seeds the final pass.
  // opts.initTrials sets the count (default 3; 1 = off).
  const finalRun = async (scale) => {
    const trials = n <= (opts.initTrialsUpTo ?? 60) ? (opts.initTrials ?? 3) : 1;   // opts.initTrialsUpTo: the set size up to which init pairs are tried (?inittrialsupto=N)
    if (trials <= 1) return runGeometry(scale, true, true);
    let bestK = 0, bestRes = null;
    for (let k = 0; k < trials; k++) {
      checkAbort();
      const r = await runGeometry(scale, false, false, k);
      if (!r) { log(`  init trial ${k}: no reconstruction`); continue; }
      log(`  init trial ${k}: ${r.cams.length}/${n} cams, median reproj ${r.medErr.toFixed(2)}px`);
      if (!bestRes || r.cams.length > bestRes.cams.length ||
          (r.cams.length === bestRes.cams.length && r.medErr < bestRes.medErr)) { bestRes = r; bestK = k; }
    }
    if (bestK) log(`  init trials: seeding the final pass from the rank-${bestK} pair`);
    return runGeometry(scale, true, true, bestK);
  };

  // known focal (sliced cubemap faces know f exactly; EXIF one day): skip
  // the search — it also only sweeps photo-like FOVs (0.78-1.56x maxDim),
  // which a 100-degree face (0.42x) sits far outside of.
  if (opts.focalPx) {
    const knownScale = opts.focalPx / (1.2 * Math.max(images[0].fw, images[0].fh));
    log(`focal known: ${opts.focalPx.toFixed(1)}px (${(knownScale * 1.2).toFixed(2)}x maxDim) — search skipped`);
    const final = await finalRun(knownScale);
    if (!final) throw new Error('known-focal reconstruction failed — need more parallax/overlap');
    log(`SfM done: ${final.cams.length}/${n} cameras registered, ${final.points.length} points, ` +
        (final.rmsBA != null ? `BA rms ${final.rmsBA.toFixed(2)}px` : `median reproj ${final.medErr.toFixed(2)}px`));
    if (final.cams.length < 2 || final.points.length < 50) {
      throw new Error('reconstruction too sparse');
    }
    Object.defineProperty(final, '_feats', { value: feats, enumerable: false });
    return final;
  }

  // focal PRIOR (EXIF 35 mm-equivalent, session.solve): trust it the way
  // COLMAP does — one registration at the prior, BA refines f — but keep the
  // search as the fallback when the prior is off (digital zoom, a crop, a
  // wrong tag): a prior that registers fewer than 60 % of the images is
  // rejected. Saves the four search registrations (~80 s on truck-251 at the
  // upsampled octave; the whole "Solving positions" beat on a phone).
  if (opts.focalPrior > 0) {
    const priorScale = opts.focalPrior / (1.2 * Math.max(images[0].fw, images[0].fh));
    const t0p = performance.now();
    log(`focal prior from EXIF: ${opts.focalPrior.toFixed(1)}px (${(priorScale * 1.2).toFixed(2)}x maxDim) — trying it before the search`);
    const final = await finalRun(priorScale);
    if (final && final.cams.length >= 0.6 * n && final.points.length >= 50) {
      log(`  prior accepted: ${final.cams.length}/${n} cameras in ${((performance.now() - t0p) / 1000).toFixed(1)}s (search skipped)`);
      log(`SfM done: ${final.cams.length}/${n} cameras registered, ${final.points.length} points, ` +
          (final.rmsBA != null ? `BA rms ${final.rmsBA.toFixed(2)}px` : `median reproj ${final.medErr.toFixed(2)}px`));
      Object.defineProperty(final, '_feats', { value: feats, enumerable: false });
      return final;
    }
    log(`  prior rejected (${final ? final.cams.length : 0}/${n} cameras) — falling back to the focal search`);
  }

  const candidates = [];
  let fi = 0;
  const focalScales = opts.focalScales || FOCAL_SCALES;   // opts.focalScales: a narrower search (EXIF-informed, or a quick tier)
  // The search only RANKS focals, so it registers a subsample: every k-th
  // photo, at most 48 (2026-09-10: on truck-251 each candidate was a 5-25 s
  // full registration; six candidates plus the edge bracket now cost less
  // than the old four). Rigs and small sets use every image.
  let searchSet = null;
  // Sets up to 120 images search on EVERY image (2026-09-15): an orbit thinned to
  // 100 frames was searched on 34 (every 3rd = every 6th captured frame), and the
  // bracket walked to 0.39x maxDim on 27/50 cameras — a wrong focal, scene radius
  // 42 instead of 5.8, the person a ghost; the same clip searched on 104 images
  // stopped at 0.69x. Above 120 the subsample stays (a 251-photo set is 5-25 s
  // per candidate at full size).
  // the focal search registers a <= 48-image subsample above 48 images (c53cc23). The avatar
  // branch raised this to 120 (868f9b0: on the pre-fix pair graph a 33-of-65 subsample picked
  // 0.44x and morphed); with the RANSAC guard and the video gate that condition is gone, and
  // the every-image search bent a Firefox solve 2.5 % off live (2026-09-17d). Back to 48;
  // opts.searchSubsetAbove for experiments (?sfmsubabove=N)
  if (opts.searchSubset !== false && !opts.rigs && n > (opts.searchSubsetAbove ?? 48)) {
    const stride = Math.ceil(n / 48);
    searchSet = new Set();
    for (let i = 0; i < n; i += stride) searchSet.add(i);
    log(`focal search on ${searchSet.size} of ${n} images (every ${stride}th)`);
  }
  let nS = searchSet ? searchSet.size : n;
  const searchPass = async () => {
    for (const s of focalScales) {
      ev({ stage: 'focal', done: fi++, total: focalScales.length, detail: { fScale: s * 1.2 } });
      checkAbort();
      const t0c = performance.now();
      const res = await runGeometry(s, false, false, 0, searchSet);
      const fEff = (s * 1.2).toFixed(2);
      const secs = ((performance.now() - t0c) / 1000).toFixed(1);
      if (!res) {
        log(`focal ${fEff}x maxDim: no valid initialization (${secs}s)`);
        continue;
      }
      res.angErr = res.medErr / (s * 1.2 * 640); // relative units; constant factor irrelevant
      res.rankErr = useGlobalSearch ? res.medErr : res.angErr;
      log(`focal ${fEff}x maxDim: ${res.cams.length}/${nS} cams, ${res.points.length} pts, ` +
          `median reproj ${res.medErr.toFixed(2)}px (angular ${(res.angErr * 1e4).toFixed(2)}e-4) in ${secs}s`);
      candidates.push(res);
      await tick();
    }
  };
  await searchPass();
  // A search that registers almost none of its subsample ranks focals on
  // noise: Lisa's 4K orbit (208 frames, every 5th) put 5-6 of 42 cameras on
  // EVERY candidate, and the pixel median picked 1.20x maxDim for a 0.69x
  // ultra-wide clip — the final pass then registered 201/208 cameras with the
  // person triangulated to 108 points (2026-09-14). Densify the subsample
  // (every 2nd frame) when the best candidate holds under a third of it.
  if (candidates.length && searchSet && Math.max(...candidates.map((r) => r.cams.length)) < nS / 3 && Math.ceil(n / 48) > 2) {
    searchSet = new Set(); for (let i = 0; i < n; i += 2) searchSet.add(i);
    nS = searchSet.size; fi = 0; candidates.length = 0;
    log(`focal search: the best candidate held under a third of the subsample — searching again on ${nS} of ${n} images (every 2nd)`);
    await searchPass();
  }
  if (!candidates.length && searchSet) {
    // a subsample too sparse to chain (a forward walk sampled every k-th
    // frame): search on every image instead
    log('focal search: no candidate initialized on the subsample — searching on every image');
    searchSet = null; nS = n; fi = 0;
    await searchPass();
  }
  if (!candidates.length) throw new Error('no focal candidate produced a valid initialization — need more parallax/overlap');
  // A candidate must register (nearly) the most cameras to compete, but among
  // those the PIXEL median error picks the winner: a too-long focal can bend
  // the reconstruction to register EVERYTHING (measured on truck at 960px
  // features: 1.56x registered 42/42 and won on count while the true 0.78x
  // sat at 37/42 with clearly better pixel error -> ATE 0.04% vs 2.28%). The
  // angular metric is no tiebreak either — it divides by f and favored the
  // same wrong candidate. The near-count window (88%) keeps the completeness
  // prior for genuinely fragile candidates; the BA rerun of the winner
  // recovers the few cameras the cheap search pass missed.
  const maxCams = Math.max(...candidates.map((r) => r.cams.length));
  const eligible = candidates.filter((r) => r.cams.length >= Math.ceil(0.88 * maxCams));
  let best = null;
  for (const res of eligible) {
    if (!best || res.medErr < best.medErr) best = res;
  }

  // Edge bracket (2026-09-10): a winner on the grid's edge says the truth may
  // lie beyond it — step outward by 12 % while a candidate keeps at least
  // 88 % of the cameras and lowers the pixel median (at most five steps:
  // 0.69 -> 0.36 reaches a phone's 0.5x ultra-wide, 1.56 -> 2.7 its 3x tele).
  // The float value itself is left to BA, which converges from within ~10 %.
  const gridWinner = best;   // the grid's pick, kept for the bracket verification below
  if (opts.focalBracket !== false && best) {
    const grid = focalScales.slice().sort((a, b) => a - b);
    const dir = best.fScale <= grid[0] ? 1 / 1.12 : best.fScale >= grid[grid.length - 1] ? 1.12 : 0;
    for (let step = 0; dir && step < 5; step++) {
      const s = best.fScale * dir;
      checkAbort();
      const t0b = performance.now();
      const res = await runGeometry(s, false, false, 0, searchSet);
      const secs = ((performance.now() - t0b) / 1000).toFixed(1);
      if (!res) { log(`focal bracket ${(s * 1.2).toFixed(2)}x maxDim: no valid initialization (${secs}s)`); break; }
      res.angErr = res.medErr / (s * 1.2 * 640);
      const better = res.cams.length >= Math.ceil(0.88 * best.cams.length) && res.medErr < best.medErr;
      log(`focal bracket ${(s * 1.2).toFixed(2)}x maxDim: ${res.cams.length}/${nS} cams, median reproj ${res.medErr.toFixed(2)}px in ${secs}s` +
          (better ? ' — better, stepping on' : ' — stop'));
      if (!better) break;
      best = res;
    }
  }

  // re-run the winner verbosely (with bundle adjustment) to produce the
  // final reconstruction
  log(`focal search winner: ${(best.fScale * 1.2).toFixed(2)}x maxDim — rerunning with BA ...`);
  const t0w = performance.now();
  let final = await finalRun(best.fScale);
  if (!final) throw new Error('focal winner failed on rerun (unexpected)');
  log(`  final registration + BA in ${((performance.now() - t0w) / 1000).toFixed(1)}s`);
  // Bracket verification (2026-09-15): the bracket's rule — more cameras and a
  // lower pixel median on the cheap pass — is satisfied by a wrong WIDE focal on
  // an orbit around a person (Lisa: 0.62x -> 0.39x on 54 -> 62 of 100 cameras,
  // scene radius 22 instead of 5.8). When the bracket moved, the grid's own
  // winner gets the same full solve and the one that registers more frames
  // stays (a tie goes to the lower BA rms).
  // Final trials (2026-09-17e). The grid ranks its focal candidates by the cheap pass's pixel
  // median, and among near-ties that ranking flips on the pixels (Chrome's decoder put 0.96x
  // first, Firefox's 0.69x; both BA to f 445 px) — but the final registration seeded from each
  // lands in a DIFFERENT optimum: 0.88 px rms and 0.5 % off the reference from one, 0.58 px and
  // 0.2 % from the other. The full-BA rms tells them apart where the cheap median cannot. So the
  // final pass runs from the top candidates (the bracket winner, the grid winner when the
  // bracket moved, the grid's runner-up) and keeps the one that registers the most cameras,
  // then the lowest BA rms. This subsumes the earlier bracket re-check (`focalVerify`). Sets
  // above 120 images run a single final pass (a final pass with BA is the expensive step).
  // opts.finalTrials (?finaltrials=N): the number of candidates; 1 = a single pass.
  {
    const trialsN = n <= 120 ? (opts.finalTrials ?? 2) : 1;
    const ranked = eligible.slice().sort((a, b) => a.medErr - b.medErr);
    const order = [best, ...(gridWinner && gridWinner !== best ? [gridWinner] : []), ...ranked];
    const tried = new Set([best.fScale]); const scales = [best.fScale];
    for (const r of order) { if (scales.length >= Math.max(trialsN, gridWinner !== best ? 2 : 1)) break; if (!tried.has(r.fScale)) { tried.add(r.fScale); scales.push(r.fScale); } }
    const rms = (r) => (r && r.rmsBA != null ? r.rmsBA : r ? r.medErr : Infinity);
    for (const sc of scales.slice(1)) {
      const t0v = performance.now();
      let alt = null;
      try { alt = await finalRun(sc); } catch (e) { log(`final trial ${(sc * 1.2).toFixed(2)}x: failed (${e.message || e})`); }
      if (!alt) continue;
      const keepAlt = alt.cams.length > final.cams.length || (alt.cams.length === final.cams.length && rms(alt) < rms(final));
      log(`final trial ${(sc * 1.2).toFixed(2)}x: ${alt.cams.length}/${n} cams, rms ${rms(alt).toFixed(2)}px against ${final.cams.length}/${n} at ${rms(final).toFixed(2)}px in ${((performance.now() - t0v) / 1000).toFixed(1)}s — ${keepAlt ? 'keeping it' : 'keeping the first'}`);
      if (keepAlt) final = alt;
    }
  }

  log(`SfM done: ${final.cams.length}/${n} cameras registered, ${final.points.length} points, ` +
      (final.rmsBA != null ? `BA rms ${final.rmsBA.toFixed(2)}px` : `median reproj ${final.medErr.toFixed(2)}px`));
  if (final.cams.length < 2 || final.points.length < 50)
    throw new Error('reconstruction too sparse to continue');
  Object.defineProperty(final, '_feats', { value: feats, enumerable: false });
  return final;
}

/** The solve, with one automatic retry: when the strict pair gate (E-inliers >= 40 %
 *  of raw matches) registers fewer than 70 % of the images, rerun the geometry
 *  (features cached, matching is seconds on the GPU) with the absolute gates —
 *  100 inliers for any pair, 15 for neighbours in capture order — and keep the
 *  pass that registered more. MEASURED 2026-09-08: a low-texture 34-photo
 *  statue goes 19 -> 24 registered; truck (251/251 on the first pass) never
 *  retries — the absolute gates as a DEFAULT perturbed its reconstruction path
 *  (250/251, 22.67 dB vs 25.72). opts.pairRelax = false disables the retry. */
export async function runSfM(images, log, sampleColor, opts = {}) {
  const first = await runSfMOnce(images, log, sampleColor, opts);
  const n = images.length;
  const relaxable = opts.pairRelax !== false && opts.pairMinInliers == null && opts.pairMinInliersAdj == null && !opts.rigs && n >= 6;
  // a video is a chain: a run of unregistered frames in capture order is a
  // broken bridge, not scattered misses — Filip's orbit registered 43/59
  // (over the 70 % bar) with frames 17-32 missing, the whole back of the
  // person (2026-09-14). The relaxed gate is what mends such a gap.
  const reg = new Set(first.cams.map((c) => c.imgIdx)); let gap = 0, run = 0;
  for (let i = 0; i < n; i++) { run = reg.has(i) ? 0 : run + 1; if (run > gap) gap = run; }
  // the gap rule (retry above 70 % when >= 5 consecutive frames are missing) is opt-in since
  // 2026-09-17d (opts.pairRelaxGap, ?pairrelax=gap): with the video gate the first pass registers
  // the chain itself, and the extra pass was one of three that bent a Firefox solve off live
  if (!relaxable || (first.cams.length >= 0.7 * n && !(opts.pairRelaxGap && gap >= 5))) return first;
  log(first.cams.length >= 0.7 * n
    ? `registration ${first.cams.length}/${n} with a gap of ${gap} consecutive frames — retrying with the absolute pair gate (100 / neighbours 15)`
    : `registration ${first.cams.length}/${n} < 70 % — retrying with the absolute pair gate (100 / neighbours 15)`);
  let second = null;
  try {
    second = await runSfMOnce(images, log, sampleColor, { ...opts, pairMinInliers: 100, pairMinInliersAdj: 15, _feats: first._feats });
  } catch (e) { log(`retry failed: ${e.message}`); }
  if (second && second.cams.length > first.cams.length) {
    log(`retry registered ${second.cams.length}/${n} (first pass ${first.cams.length}) — keeping the retry`);
    return second;
  }
  log(`retry registered ${second ? second.cams.length : 0}/${n} — keeping the first pass`);
  return first;
}
