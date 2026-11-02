// io/video.js — turn a video file into training photographs, in the browser.
//
// Users film places; they rarely have photo sets. v2 (2026-09-08) follows what
// the 3DGS dataset tools converged on (Reflct Sharp Frames, SLAM keyframing):
//
//   decode   EVERY frame through WebCodecs (Mediabunny demux + VideoDecoder,
//            hardware, exact timestamps, rotation metadata honoured) — not
//            "whatever a 3x playback happened to present"
//   score    sharp-frames' resolution-normalised hybrid focus metric on a
//            512-px grayscale: Gaussian 5x5 denoise, then Laplacian variance
//            and Tenengrad (mean Sobel energy), log-combined 50/50; plus
//            luminance mean / clipped fraction (exposure outliers) and a
//            motion proxy (projection cross-correlation shift vs the previous
//            frame) so a paused camera does not yield 40 identical frames
//   select   local outlier removal (a frame that dips below its neighbours is
//            motion blur — a global threshold would just prefer textured
//            frames), then MOTION windows: a window closes when the camera has
//            moved ~10 % of the frame width (90 % overlap; the Reflct 80 % guide
//            registered 27/96 on a close phone orbit, 90 % registered 176/181),
//            bounded to [0.15 s, 0.25 s]; the sharpest survivor of each window
//            is kept; a device cap widens windows rather than dropping the
//            sharpest
//   capture  decode only the winners again at full resolution (no seeking,
//            no duplicate decodes) and encode JPEG q0.95
//
// Falls back to the <video>/requestVideoFrameCallback scan when WebCodecs or
// the codec is unavailable (same scorer and selector, ~10 samples/s).
//
// Mediabunny (MPL-2.0, Vanilagy) is vendored as src/vendor/mediabunny.min.mjs
// and imported lazily — the app should not pay 660 KB for photo sets.

/**
 * @typedef {object} VideoExtractOptions
 * @property {number} [maxFrames]          device cap; default 300 desktop / 140 mobile
 * @property {number} [minFrames=24]
 * @property {number} [overlap=0.9]        target overlap between kept frames (motion budget = 1 - overlap of the width).
 *   MEASURED 2026-09-08 on a close-range phone orbit (LisaAvatar): 0.8 → 96 frames, 27 registered;
 *   0.9 → 181 frames, 176 registered. Registration needs parallax density, not just overlap.
 * @property {number} [minGapSec=0.15]     a window never closes faster than this
 * @property {number} [maxGapSec=0.25]     nor later. MEASURED 2026-09-10 on a phone walk (camping.mov, 15 fixed test
 *   photos, 30k): the motion proxy under-reads forward motion so this ceiling paces walks — 1.0 s gave 43 training
 *   frames and 22.8 dB, 0.35 s 72 / 24.1, 0.25 s 88 / 25.3, 0.15 s 126 / 24.7. Orbits close windows by motion first.
 * @property {number} [outlierWindow=15]   local window for blur-dip removal (frames)
 * @property {number} [outlierSensitivity=0.6]  0 = off; a frame is dropped when its focus < (1 - s·0.5) × neighbour median
 * @property {number} [jpegQuality=0.95]
 * @property {'auto'|'webcodecs'|'element'} [engine='auto']
 * @property {'longest'|'all'} [shots='longest'] edited clips: reconstruct one continuous take (default) or keep every shot
 * @property {'sharp'|'uniform'} [pick='sharp']  'uniform' takes frames at a fixed rate (`uniformFps`) with no sharpness selection — a control for the selector
 * @property {number} [uniformFps=3]
 * @property {boolean|{width?: number, count?: number}} [thumbs]  keep small thumbnails from the scan (`count` spread over the
 *   video, default 360 at 96 px) for a review UI; returned as `thumbs: [{t, index, canvas}]`
 * @property {(ctx: VideoReview) => Promise<{picks?: number[]}|null>} [review]  called between the scan and the
 *   capture with the scores, the automatic picks and a `plan(range)` re-selector; return the picks to capture,
 *   or null to cancel. Without it the automatic picks are captured.
 * @property {number[]} [forceTimes]      timestamps (s) whose nearest frames are always kept (a fixed evaluation set across extraction variants)
 * @property {number} [forceExcludeSec=0] no other pick within this distance of a forced time (keeps a near-duplicate training view out of the test)
 * @property {(e: {stage: 'scan'|'capture', done: number, total: number}) => void} [onProgress]
 * @property {(msg: string) => void} [log]
 */

const ANALYSIS_LONG_EDGE = 512;
const MOTION_EDGE = 256;
const MOTION_BLOCKS = 4;      // 4x4 block flow: forward motion shows as radial block shifts, a pan as a common one

// ---------------------------------------------------------------- scoring

/** 5x5 Gaussian (sigma 1) on a grayscale float buffer, separable. */
function gauss5(src, w, h, tmp, dst) {
  const k = [0.0614, 0.2448, 0.3877, 0.2448, 0.0614]; // sigma 1.0, normalised
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let i = -2; i <= 2; i++) {
        const xx = Math.min(w - 1, Math.max(0, x + i));
        s += k[i + 2] * src[row + xx];
      }
      tmp[row + x] = s;
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let i = -2; i <= 2; i++) {
        const yy = Math.min(h - 1, Math.max(0, y + i));
        s += k[i + 2] * tmp[yy * w + x];
      }
      dst[y * w + x] = s;
    }
  }
}

/** sharp-frames `normalized_laplacian_tenengrad_v1` on a denoised gray buffer
 *  (values 0..255): expm1(0.5·log1p(LaplacianVar) + 0.5·log1p(mean Sobel²)). */
function focusScore(g, w, h) {
  let lsum = 0, lsq = 0, ten = 0;
  const n = (w - 2) * (h - 2);
  for (let y = 1; y < h - 1; y++) {
    const r = y * w;
    for (let x = 1; x < w - 1; x++) {
      const i = r + x;
      const lap = 4 * g[i] - g[i - 1] - g[i + 1] - g[i - w] - g[i + w];
      lsum += lap; lsq += lap * lap;
      const gx = (g[i - w + 1] + 2 * g[i + 1] + g[i + w + 1]) - (g[i - w - 1] + 2 * g[i - 1] + g[i + w - 1]);
      const gy = (g[i + w - 1] + 2 * g[i + w] + g[i + w + 1]) - (g[i - w - 1] + 2 * g[i - w] + g[i - w + 1]);
      ten += gx * gx + gy * gy;
    }
  }
  const lmean = lsum / n;
  const lapVar = Math.max(0, lsq / n - lmean * lmean);
  const tenengrad = ten / n;
  return { focus: Math.expm1(0.5 * Math.log1p(lapVar) + 0.5 * Math.log1p(tenengrad)), lapVar, tenengrad };
}

/** Exposure statistics on the gray buffer: mean and clipped fraction. */
function exposure(g, n) {
  let sum = 0, clipped = 0;
  for (let i = 0; i < n; i++) { const v = g[i]; sum += v; if (v <= 4 || v >= 251) clipped++; }
  return { lum: sum / n / 255, clip: clipped / n };
}

/** Sub-pixel shift between two 1-D projections (SSE search ± maxShift, parabolic refinement). */
function projShift(a, b, n, maxShift) {
  const errs = new Float64Array(2 * maxShift + 1);
  let bestS = 0, bestE = Infinity;
  for (let s = -maxShift; s <= maxShift; s++) {
    let e = 0, c = 0;
    for (let i = Math.max(0, -s); i < Math.min(n, n - s); i++) { const d = a[i] - b[i + s]; e += d * d; c++; }
    e /= Math.max(1, c);
    errs[s + maxShift] = e;
    if (e < bestE) { bestE = e; bestS = s; }
  }
  const k = bestS + maxShift;
  if (k > 0 && k < 2 * maxShift) {
    const em = errs[k - 1], e0 = errs[k], ep = errs[k + 1];
    const den = em - 2 * e0 + ep;
    if (den > 1e-9) return bestS + 0.5 * (em - ep) / den;
  }
  return bestS;
}

/** Camera motion between consecutive small gray frames as a fraction of the
 *  width: median magnitude of per-block projection shifts on a 4x4 grid
 *  (sub-pixel). A pan moves every block the same way, walking forward moves
 *  them radially — both count; a paused camera measures ~0. At 30 fps and
 *  256 px this resolves the 0.3–1 px/frame of a walking capture. */
function motionShift(prev, cur, w, h, maxShift = 12) {
  const bw = Math.floor(w / MOTION_BLOCKS), bh = Math.floor(h / MOTION_BLOCKS);
  const mags = [];
  const px = new Float64Array(bw), cx = new Float64Array(bw), py = new Float64Array(bh), cy = new Float64Array(bh);
  for (let by = 0; by < MOTION_BLOCKS; by++) for (let bx = 0; bx < MOTION_BLOCKS; bx++) {
    px.fill(0); cx.fill(0); py.fill(0); cy.fill(0);
    for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) {
      const i = (by * bh + y) * w + bx * bw + x;
      px[x] += prev[i]; cx[x] += cur[i]; py[y] += prev[i]; cy[y] += cur[i];
    }
    const dx = projShift(px, cx, bw, maxShift), dy = projShift(py, cy, bh, maxShift);
    mags.push(Math.hypot(dx, dy));
  }
  mags.sort((a, b) => a - b);
  return mags[mags.length >> 1] / w;
}

/** Analyse one drawn frame: fills `gray` from the canvas, returns the record. */
function makeAnalyzer(sw, sh) {
  const n = sw * sh;
  const gray = new Float32Array(n), tmp = new Float32Array(n), den = new Float32Array(n);
  const mw = MOTION_EDGE, mh = Math.max(2, Math.round(mw * sh / sw));
  let prevSmall = null;
  const small = new Float32Array(mw * mh);
  return {
    analyze(ctx, t) {
      const d = ctx.getImageData(0, 0, sw, sh).data;
      for (let i = 0; i < n; i++) gray[i] = 0.299 * d[i * 4] + 0.587 * d[i * 4 + 1] + 0.114 * d[i * 4 + 2];
      gauss5(gray, sw, sh, tmp, den);
      const f = focusScore(den, sw, sh);
      const e = exposure(gray, n);
      // motion proxy on a 128-px box-downsample of the denoised gray
      const fx = sw / mw, fy = sh / mh;
      for (let y = 0; y < mh; y++) for (let x = 0; x < mw; x++) {
        const sx = Math.min(sw - 1, Math.floor(x * fx)), sy = Math.min(sh - 1, Math.floor(y * fy));
        small[y * mw + x] = den[sy * sw + sx];
      }
      const motion = prevSmall ? motionShift(prevSmall, small, mw, mh) : 0;
      // frame difference (0..1) vs the previous frame — a hard cut is a spike far
      // above the rolling level of a moving camera
      let diff = 0;
      if (prevSmall) { let d = 0; for (let i = 0; i < small.length; i++) d += Math.abs(small[i] - prevSmall[i]); diff = d / small.length / 255; }
      prevSmall = prevSmall || new Float32Array(mw * mh);
      prevSmall.set(small);
      return { t, focus: f.focus, lapVar: f.lapVar, tenengrad: f.tenengrad, lum: e.lum, clip: e.clip, motion, diff };
    },
  };
}

// -------------------------------------------------------------- selection

/** Mark blur dips: a frame whose focus falls below (1 - 0.5·s) × the median of
 *  its neighbours (window w) is an outlier. Mirrors sharp-frames' intent:
 *  compare to neighbours, never to a global threshold. */
function markOutliers(frames, win = 15, s = 0.6) {
  const half = Math.max(2, win >> 1);
  const n = frames.length;
  for (let i = 0; i < n; i++) {
    if (s <= 0) { frames[i].blur = false; continue; }
    const nb = [];
    for (let j = Math.max(0, i - half); j <= Math.min(n - 1, i + half); j++) if (j !== i) nb.push(frames[j].focus);
    if (nb.length < 4) { frames[i].blur = false; continue; }
    nb.sort((a, b) => a - b);
    const med = nb[nb.length >> 1];
    frames[i].blur = frames[i].focus < (1 - 0.5 * s) * med;
  }
  return frames;
}

/** Motion windows: accumulate the motion proxy; close a window when the camera
 *  moved `budget` of the frame width (or maxGap elapsed), never before minGap.
 *  Keep the sharpest non-blurred frame of each window. */
function selectByMotion(frames, { budget, minGap, maxGap }) {
  const picks = [];
  let start = 0, acc = 0;
  const flush = (end) => {
    let best = -1;
    for (let i = start; i <= end; i++) {
      if (frames[i].blur) continue;
      if (best < 0 || frames[i].focus > frames[best].focus) best = i;
    }
    if (best < 0) { // every frame in the window was a blur dip — take the least bad
      for (let i = start; i <= end; i++) if (best < 0 || frames[i].focus > frames[best].focus) best = i;
    }
    if (best >= 0) picks.push(best);
  };
  for (let i = 1; i < frames.length; i++) {
    acc += frames[i].motion;
    const dt = frames[i].t - frames[start].t;
    if ((acc >= budget && dt >= minGap) || dt >= maxGap) { flush(i); start = i + 1; acc = 0; }
  }
  if (start < frames.length) flush(frames.length - 1);
  return picks;
}

/** Shot boundaries: a frame whose difference to its predecessor exceeds both an
 *  absolute floor and k× the rolling median of the neighbourhood is a cut (an
 *  edited clip, a drone montage, a phone that stopped and restarted). Returns
 *  [{ start, end }] index ranges (inclusive). */
export function detectShots(frames, { absMin = 0.12, ratio = 6, win = 30 } = {}) {
  const n = frames.length;
  const cuts = [];
  for (let i = 1; i < n; i++) {
    const d = frames[i].diff || 0;
    if (d < absMin) continue;
    const nb = [];
    for (let j = Math.max(1, i - win); j <= Math.min(n - 1, i + win); j++) if (j !== i) nb.push(frames[j].diff || 0);
    nb.sort((a, b) => a - b);
    const med = nb.length ? nb[nb.length >> 1] : 0;
    if (d > ratio * Math.max(med, 0.005)) cuts.push(i);
  }
  const shots = [];
  let start = 0;
  for (const c of cuts) { shots.push({ start, end: c - 1 }); start = c; }
  shots.push({ start, end: n - 1 });
  return shots.filter((sh) => sh.end >= sh.start);
}

/** Full selection with the device cap / floor applied by rescaling the budget.
 *  opts.shots: 'longest' (default — one continuous take is what a
 *  reconstruction needs) or 'all' (windows still never span a cut). */
export function selectFrames(frames, opts = {}) {
  const shots = detectShots(frames);
  for (const f of frames) f.picked = false;
  if (shots.length > 1 && (opts.shots ?? 'longest') === 'longest') {
    let best = shots[0];
    for (const sh of shots) if (frames[sh.end].t - frames[sh.start].t > frames[best.end].t - frames[best.start].t) best = sh;
    const sub = frames.slice(best.start, best.end + 1);
    const r = selectFramesContinuous(sub, opts);
    return { picks: r.picks.map((i) => i + best.start), budget: r.budget, shots, shot: best };
  }
  if (shots.length > 1) {
    let picks = [], budget = 0;
    for (const sh of shots) {
      const sub = frames.slice(sh.start, sh.end + 1);
      if (sub.length < 3) continue;
      const r = selectFramesContinuous(sub, { ...opts, maxFrames: Math.max(3, Math.round((opts.maxFrames ?? defaultMaxFrames()) * sub.length / frames.length)), minFrames: 3 });
      picks = picks.concat(r.picks.map((i) => i + sh.start)); budget = r.budget;
    }
    return { picks, budget, shots, shot: null };
  }
  const r = selectFramesContinuous(frames, opts);
  return { ...r, shots, shot: shots[0] };
}

/** Forced timestamps (opts.forceTimes) join the picks: the nearest scored
 *  frame to each, inserted in time order, never duplicated; other picks within
 *  opts.forceExcludeSec of a forced time are dropped. */
export function applyForced(frames, picks, opts) {
  if (!opts.forceTimes || !opts.forceTimes.length) return;
  const excl = opts.forceExcludeSec || 0;
  const set = new Set(excl > 0 ? picks.filter((i) => opts.forceTimes.every((t) => Math.abs(frames[i].t - t) >= excl)) : picks);
  for (const t of opts.forceTimes) {
    let best = 0;
    for (let i = 1; i < frames.length; i++) if (Math.abs(frames[i].t - t) < Math.abs(frames[best].t - t)) best = i;
    set.add(best);
  }
  picks.length = 0;
  for (const i of [...set].sort((a, b) => a - b)) picks.push(i);
  for (let i = 0; i < frames.length; i++) frames[i].picked = set.has(i);
}

/**
 * @typedef {object} VideoReview
 * @property {Array<object>} frames   every scored frame ({t, focus, motion, blur, diff, picked})
 * @property {Array<{start: number, end: number}>} shots
 * @property {number[]} picks         the automatic picks (frame indices)
 * @property {number} budget          the motion budget those picks used
 * @property {{start: number, end: number}|null} suggested  a time span for long videos (null = whole)
 * @property {Array<{t: number, index: number, canvas: *}>} thumbs
 * @property {(range: {start: number, end: number}|null) => {picks: number[], budget: number}} plan
 *   re-run the selection on a time range (pure, instant — the scores are kept)
 * @property {number} cap  the device cap on frames
 */

/** Selection restricted to a time range [start, end] (s); indices are into
 *  the full `frames` array. Pure: repeatable from a UI slider. */
export function planSelection(frames, opts = {}, range = null) {
  let lo = 0, hi = frames.length - 1;
  if (range) {
    while (lo < hi && frames[lo].t < range.start) lo++;
    while (hi > lo && frames[hi].t > range.end) hi--;
  }
  if (hi - lo < 2) { lo = Math.max(0, Math.min(lo, frames.length - 3)); hi = Math.min(frames.length - 1, lo + 2); }
  const sub = frames.slice(lo, hi + 1);
  const r = selectFrames(sub, opts);
  const picks = r.picks.map((i) => i + lo);
  applyForced(frames, picks, opts);
  for (const f of frames) f.picked = false;
  for (const i of picks) frames[i].picked = true;
  return {
    picks, budget: r.budget,
    shots: r.shots.map((sh) => ({ start: sh.start + lo, end: sh.end + lo })),
    shot: r.shot ? { start: r.shot.start + lo, end: r.shot.end + lo } : null,
    range: { start: frames[lo].t, end: frames[hi].t },
  };
}

/** A long video: the span the device cap can hold at the natural density
 *  (no widened windows), placed where the footage is sharpest. Null when the
 *  whole take fits. */
export function suggestSpan(frames, opts = {}) {
  const cap = opts.maxFrames ?? defaultMaxFrames();
  const shots = detectShots(frames);
  let best = shots[0];
  for (const sh of shots) if (frames[sh.end].t - frames[sh.start].t > frames[best.end].t - frames[best.start].t) best = sh;
  const sub = frames.slice(best.start, best.end + 1);
  markOutliers(sub, opts.outlierWindow ?? 15, opts.outlierSensitivity ?? 0.6);
  const natural = selectByMotion(sub, { budget: 1 - (opts.overlap ?? 0.9), minGap: opts.minGapSec ?? 0.15, maxGap: opts.maxGapSec ?? 0.25 });
  if (natural.length <= cap) return null;
  // window of `cap` consecutive natural picks with the highest mean focus
  const pre = [0];
  for (const i of natural) pre.push(pre[pre.length - 1] + sub[i].focus);
  let bj = 0, bs = -Infinity;
  for (let j = 0; j + cap <= natural.length; j++) {
    const m = pre[j + cap] - pre[j];
    if (m > bs) { bs = m; bj = j; }
  }
  return { start: sub[natural[bj]].t, end: sub[natural[bj + cap - 1]].t };
}

/** Scan-time thumbnails: how many and how wide. */
function thumbPlan(opts, total, sw, sh) {
  if (!opts.thumbs) return null;
  const o = typeof opts.thumbs === 'object' ? opts.thumbs : {};
  const count = o.count || 360, tw = o.width || 96;
  return { every: Math.max(1, Math.ceil(total / count)), tw, th: Math.max(2, Math.round(tw * sh / sw)) };
}

/** The review hook between scan and capture (shared by both engines). */
async function reviewPicks(frames, sel, opts, extra) {
  if (!opts.review) return sel.picks;
  const cap = opts.maxFrames ?? defaultMaxFrames();
  const r = await opts.review({
    frames, shots: sel.shots, shot: sel.shot, picks: sel.picks, budget: sel.budget, cap,
    suggested: suggestSpan(frames, opts),
    plan: (range) => planSelection(frames, opts, range),
    ...extra,
  });
  if (r === null) throw new Error('cancelled');
  return (r && r.picks) ? r.picks : sel.picks;
}

/** Selection on one continuous take. */
function selectFramesContinuous(frames, opts = {}) {
  if (opts.pick === 'uniform') {   // control: a fixed rate, no scoring
    const step = 1 / (opts.uniformFps || 3);
    const picks = [];
    let next = frames[0].t;
    for (let i = 0; i < frames.length; i++) {
      if (frames[i].t + 1e-6 >= next) { picks.push(i); next = frames[i].t + step; }
    }
    const set = new Set(picks);
    for (let i = 0; i < frames.length; i++) { frames[i].blur = false; frames[i].picked = set.has(i); }
    return { picks, budget: 0 };
  }
  const maxFrames = opts.maxFrames ?? defaultMaxFrames();
  const minFrames = Math.min(maxFrames, opts.minFrames ?? 24);
  const minGap = opts.minGapSec ?? 0.15, maxGap = opts.maxGapSec ?? 0.25;
  markOutliers(frames, opts.outlierWindow ?? 15, opts.outlierSensitivity ?? 0.6);
  let budget = 1 - (opts.overlap ?? 0.9);
  let picks = selectByMotion(frames, { budget, minGap, maxGap });
  // too many: widen the motion budget (fewer, better-spread windows)
  for (let k = 0; k < 12 && picks.length > maxFrames; k++) {
    budget *= 1.25;
    picks = selectByMotion(frames, { budget, minGap, maxGap });
  }
  // still too many (a fast pan against a hard cap): keep the sharpest with even spacing
  if (picks.length > maxFrames) {
    const stride = picks.length / maxFrames;
    picks = Array.from({ length: maxFrames }, (_, i) => picks[Math.floor(i * stride)]);
  }
  // too few (a slow/static video): shrink the budget and the time floor
  for (let k = 0; k < 8 && picks.length < Math.min(minFrames, frames.length) ; k++) {
    budget *= 0.7;
    picks = selectByMotion(frames, { budget, minGap: Math.max(0.1, minGap * 0.7 ** (k + 1)), maxGap: maxGap * 0.7 ** (k + 1) });
  }
  const set = new Set(picks);
  for (let i = 0; i < frames.length; i++) frames[i].picked = set.has(i);
  return { picks, budget };
}

function defaultMaxFrames() {
  const mobile = (navigator.userAgentData && navigator.userAgentData.mobile) ||
    /Android|iPhone|iPad/i.test(navigator.userAgent || '');
  return mobile ? 140 : 300;
}

// ---------------------------------------------------------------- decode

const until = (el, ev, err = 'error') => new Promise((res, rej) => {
  const ok = () => { cleanup(); res(); };
  const bad = (e) => { cleanup(); rej(new Error(`video ${err}: ${(e && e.message) || 'decode failed'}`)); };
  const cleanup = () => { el.removeEventListener(ev, ok); el.removeEventListener('error', bad); };
  el.addEventListener(ev, ok, { once: true });
  el.addEventListener('error', bad, { once: true });
});

function mkCanvas(w, h) {
  if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(w, h);
  const cv = document.createElement('canvas');
  cv.width = w; cv.height = h;
  return cv;
}

function toBlob(cv, quality) {
  if (cv.convertToBlob) return cv.convertToBlob({ type: 'image/jpeg', quality });
  return new Promise((res, rej) => cv.toBlob(
    (b) => (b ? res(b) : rej(new Error('jpeg encode failed'))), 'image/jpeg', quality));
}

const analysisSize = (w, h) => {
  const s = ANALYSIS_LONG_EDGE / Math.max(w, h);
  return [Math.max(2, Math.round(w * s)), Math.max(2, Math.round(h * s))];
};

let MBmod = null;
async function mediabunny() {
  if (!MBmod) MBmod = await import('../vendor/mediabunny.min.mjs');
  return MBmod;
}

/** Draw a VideoFrame into a canvas of the DISPLAY size (cw x ch), applying
 *  the container's rotation (clockwise degrees, phones shoot portrait). */
function drawRotated(ctx, frame, cw, ch, rot) {
  ctx.save();
  if (rot === 90) { ctx.translate(cw, 0); ctx.rotate(Math.PI / 2); ctx.drawImage(frame, 0, 0, ch, cw); }
  else if (rot === 180) { ctx.translate(cw, ch); ctx.rotate(Math.PI); ctx.drawImage(frame, 0, 0, cw, ch); }
  else if (rot === 270) { ctx.translate(0, ch); ctx.rotate(-Math.PI / 2); ctx.drawImage(frame, 0, 0, ch, cw); }
  else ctx.drawImage(frame, 0, 0, cw, ch);
  ctx.restore();
}

/** Run a VideoDecoder over the track's packets from `startPacket` (decode
 *  order, Mediabunny demux) and hand every output frame, in presentation
 *  order with the DECODER's timestamp, to onFrame(frame) — return false to
 *  stop early. The frames are closed here.
 *
 *  Why not Mediabunny's sinks: on field-coded streams (two packets per
 *  picture, e.g. a 25p camera file tagged 50 fps) the decoder emits one frame
 *  per packet pair with the right timestamps, but the sinks restamp outputs
 *  sequentially from the packet list, so the second half of the take
 *  collapses onto the first (skulli.mp4, 2026-09-11: 4628 packets, 2314
 *  frames, "ends" at 46 of 93 s). */
async function runDecoder(MB, track, startPacket, onFrame) {
  const cfg = await track.getDecoderConfig();
  const queue = [];
  let err = null, stop = false, pumping = null;
  // outputs are hardware-backed and few: close them as they arrive (a pump
  // driven from the output callback), never hold them until the next packet —
  // a decoder waiting for its frame pool never finishes flush()
  const pump = () => pumping || (pumping = (async () => {
    try {
      while (queue.length) {
        const f = queue.shift();
        try { if (!stop && (await onFrame(f)) === false) stop = true; }
        catch (e) { err = err || e; stop = true; }
        finally { f.close(); }
      }
    } finally { pumping = null; }
  })());
  const dec = new VideoDecoder({ output: (f) => { queue.push(f); pump(); }, error: (e) => { err = e; } });
  dec.configure(cfg);
  const sink = new MB.EncodedPacketSink(track);
  try {
    for await (const p of sink.packets(startPacket)) {
      if (err) throw err;
      if (stop) break;
      dec.decode(p.toEncodedVideoChunk());
      while (dec.decodeQueueSize > 16 && !err && !stop) await new Promise((r) => setTimeout(r, 0));
    }
    if (!stop && !err) await dec.flush();
    await pump();
    if (err) throw err;
  } finally {
    try { dec.close(); } catch {}
    while (queue.length) queue.shift().close();
  }
}

/** WebCodecs path: every frame scored, winners re-decoded at full size. */
async function extractWebCodecs(file, opts, log, onProgress) {
  const MB = await mediabunny();
  const input = new MB.Input({ formats: MB.ALL_FORMATS, source: new MB.BlobSource(file) });
  const track = await input.getPrimaryVideoTrack();
  if (!track) throw new Error('no video track');
  if (!(await track.canDecode())) throw new Error('codec not decodable here');
  const rotation = await track.getRotation();
  const duration = await track.computeDuration();
  const stats = await track.computePacketStats(Infinity).catch(() => null);
  const total = stats ? stats.packetCount : Math.round(duration * 30);
  let fps = stats ? stats.averagePacketRate : null;
  let colour = '';
  try { const cs = await track.getColorSpace(); if (cs && cs.transfer) colour = ` ${cs.transfer}${cs.primaries ? '/' + cs.primaries : ''}`; } catch {}

  // ---- pass 1: score every frame on the 512-px analysis canvas ----
  // display size comes from the first decoded frame (coded size, rotated)
  let W = 0, H = 0, sw = 0, sh = 0, an = null, ctx = null, tp = null;
  const frames = [];
  const thumbs = [];
  let done = 0;
  const first = (frame) => {
    const fw = frame.displayWidth, fh = frame.displayHeight;
    [W, H] = (rotation === 90 || rotation === 270) ? [fh, fw] : [fw, fh];
    [sw, sh] = analysisSize(W, H);
    an = makeAnalyzer(sw, sh);
    ctx = mkCanvas(sw, sh).getContext('2d', { willReadFrequently: true });
    tp = thumbPlan(opts, total, sw, sh);
    log(`video: ${W}x${H}${rotation ? ` (rotated ${rotation}°)` : ''}, ${duration.toFixed(1)}s, ${total} packets${fps ? ` @ ${fps.toFixed(2)}/s` : ''}${colour}`);
  };
  let lastTick = 0;
  const scanOne = (frame) => {
    if (!an) first(frame);
    drawRotated(ctx, frame, sw, sh, rotation);
    const t = frame.timestamp / 1e6;
    frames.push(an.analyze(ctx, t));
    if (tp && done % tp.every === 0) {
      const tc = mkCanvas(tp.tw, tp.th);
      tc.getContext('2d').drawImage(ctx.canvas, 0, 0, tp.tw, tp.th);
      thumbs.push({ t, index: frames.length - 1, canvas: tc });
    }
    done++;
    if (done % 8 === 0) onProgress({ stage: 'scan', done: Math.min(done, total), total });
  };
  const psink = new MB.EncodedPacketSink(track);
  const firstPacket = await psink.getFirstPacket();
  await runDecoder(MB, track, firstPacket, scanOne);
  onProgress({ stage: 'scan', done: frames.length, total: frames.length });
  if (frames.length < 2) throw new Error('could not decode frames from this video');
  if (frames.length !== total) {
    fps = frames.length / Math.max(1e-3, frames[frames.length - 1].t - frames[0].t);
    log(`${frames.length} frames from ${total} packets (field-coded or dropped) — ${fps.toFixed(2)} fps`);
  }

  // ---- selection ----
  const sel = selectFrames(frames, opts);
  applyForced(frames, sel.picks, opts);
  const { budget, shots, shot } = sel;
  const blurred = frames.filter((f) => f.blur).length;
  if (shots.length > 1) log(`${shots.length} shots (cuts at ${shots.slice(1).map((sh) => frames[sh.start].t.toFixed(1) + 's').join(', ')})${shot ? ` — keeping the longest: ${frames[shot.start].t.toFixed(1)}–${frames[shot.end].t.toFixed(1)} s` : ' — keeping all'}`);
  log(`scored ${frames.length} frames (${blurred} blur dips); kept ${sel.picks.length} — motion budget ${(budget * 100).toFixed(0)} % of the width, focus median ${median(frames.map((f) => f.focus)).toFixed(0)}`);
  const picks = await reviewPicks(frames, sel, opts, { duration, fps, videoW: W, videoH: H, thumbs });
  if (picks !== sel.picks) log(`review: capturing ${picks.length} frames`);

  // ---- pass 2: full-resolution capture of the winners ----
  // one continuous decode from the key packet before the first winner; each
  // output frame is matched to the next target by timestamp
  const times = picks.map((i) => frames[i].t).sort((x, y) => x - y);
  const half = 0.5 * (frames.length > 1 ? (frames[frames.length - 1].t - frames[0].t) / (frames.length - 1) : 0.02);
  const capCv = mkCanvas(W, H);
  const capCtx = capCv.getContext('2d');
  const out = [];
  const blobs = [];
  let ti = 0;
  if (times.length) {
    const key = (await psink.getKeyPacket(times[0])) || firstPacket;
    await runDecoder(MB, track, key, async (frame) => {
      const t = frame.timestamp / 1e6;
      while (ti < times.length && times[ti] < t - half) ti++;   // a target the decoder skipped
      if (ti >= times.length) return false;
      if (Math.abs(t - times[ti]) <= half) {
        drawRotated(capCtx, frame, W, H, rotation);
        blobs.push({ blob: await toBlob(capCv, opts.jpegQuality ?? 0.95), t });
        ti++;
        onProgress({ stage: 'capture', done: ti, total: times.length });
        if (ti >= times.length) return false;
      }
      return true;
    });
  }
  for (const { blob, t } of blobs) out.push({ source: blob, name: `frame_${String(out.length + 1).padStart(5, '0')}.jpg`, t });
  if (out.length < times.length) log(`captured ${out.length} of ${times.length} winners (decoder skipped the rest)`);
  return { frames: out, duration, sampled: frames.length, videoW: W, videoH: H, fps, rotation, engine: 'webcodecs', analysis: frames, shots, shot, thumbs };
}

/** Fallback: <video> element scan at ~10 samples/s, same scorer/selector. */
async function extractElement(file, opts, log, onProgress) {
  const sps = 10;
  const url = URL.createObjectURL(file);
  const video = document.createElement('video');
  video.muted = true; video.playsInline = true; video.preload = 'auto'; video.src = url;
  try {
    await until(video, 'loadedmetadata');
    if (!isFinite(video.duration)) {
      video.currentTime = 1e9; await until(video, 'seeked');
      video.currentTime = 0; await until(video, 'seeked');
    }
    const duration = video.duration;
    const vw = video.videoWidth, vh = video.videoHeight;
    if (!vw || !vh || !isFinite(duration) || duration <= 0) throw new Error('video has no decodable track');
    log(`video (element path): ${vw}x${vh}, ${duration.toFixed(1)}s`);
    const [sw, sh] = analysisSize(vw, vh);
    const scanCv = mkCanvas(sw, sh);
    const scanCtx = scanCv.getContext('2d', { willReadFrequently: true });
    const an = makeAnalyzer(sw, sh);
    const frames = [];
    const totalSamples = Math.max(2, Math.floor(duration * sps));
    const tp = thumbPlan(opts, totalSamples, sw, sh);
    const thumbs = [];
    const scoreNow = (t) => {
      scanCtx.drawImage(video, 0, 0, sw, sh);
      frames.push(an.analyze(scanCtx, t));
      if (tp && (frames.length - 1) % tp.every === 0) {
        const tc = mkCanvas(tp.tw, tp.th);
        tc.getContext('2d').drawImage(scanCv, 0, 0, tp.tw, tp.th);
        thumbs.push({ t, index: frames.length - 1, canvas: tc });
      }
    };
    if (typeof video.requestVideoFrameCallback === 'function') {
      video.playbackRate = 3;
      let lastT = -1, finished = false;
      const onFrame = (_now, meta) => {
        if (finished) return;
        const t = meta.mediaTime;
        if (t - lastT >= 1 / sps - 1e-3) { lastT = t; scoreNow(t); onProgress({ stage: 'scan', done: Math.min(frames.length, totalSamples), total: totalSamples }); }
        video.requestVideoFrameCallback(onFrame);
      };
      video.requestVideoFrameCallback(onFrame);
      await video.play();
      await until(video, 'ended', 'playback error');
      finished = true; video.pause();
    } else {
      for (let k = 0; k < totalSamples; k++) {
        video.currentTime = Math.min(duration - 0.001, k / sps);
        await until(video, 'seeked');
        scoreNow(video.currentTime);
        onProgress({ stage: 'scan', done: k + 1, total: totalSamples });
      }
    }
    if (frames.length < 2) throw new Error('could not decode frames from this video');
    const sel = selectFrames(frames, opts);
    applyForced(frames, sel.picks, opts);
    log(`scored ${frames.length} samples; kept ${sel.picks.length} (motion budget ${(sel.budget * 100).toFixed(0)} %)`);
    const picks = await reviewPicks(frames, sel, opts, { duration, fps: null, videoW: vw, videoH: vh, thumbs });
    const capCv = mkCanvas(vw, vh);
    const capCtx = capCv.getContext('2d');
    const out = [];
    for (let i = 0; i < picks.length; i++) {
      video.currentTime = frames[picks[i]].t;
      await until(video, 'seeked');
      capCtx.drawImage(video, 0, 0, vw, vh);
      const blob = await toBlob(capCv, opts.jpegQuality ?? 0.95);
      out.push({ source: blob, name: `frame_${String(i + 1).padStart(5, '0')}.jpg`, t: frames[picks[i]].t });
      onProgress({ stage: 'capture', done: i + 1, total: picks.length });
    }
    return { frames: out, duration, sampled: frames.length, videoW: vw, videoH: vh, fps: null, rotation: 0, engine: 'element', analysis: frames, thumbs };
  } finally {
    video.removeAttribute('src'); video.load(); URL.revokeObjectURL(url);
  }
}

const median = (a) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[s.length >> 1] : 0; };

/**
 * @param {File|Blob} file
 * @param {VideoExtractOptions} [opts]
 * @returns {Promise<{frames: Array<{source: Blob, name: string, t: number}>, duration: number,
 *   sampled: number, videoW: number, videoH: number, fps: number|null, rotation: number,
 *   engine: 'webcodecs'|'element', analysis: Array<object>}>}
 */
export async function extractSharpFrames(file, opts = {}) {
  const log = opts.log || (() => {});
  const onProgress = opts.onProgress || (() => {});
  const engine = opts.engine || 'auto';
  if (engine !== 'element' && typeof VideoDecoder !== 'undefined') {
    try {
      return await extractWebCodecs(file, opts, log, onProgress);
    } catch (e) {
      if (engine === 'webcodecs' || e.message === 'cancelled') throw e;   // a cancelled review is not a decoder failure
      log(`WebCodecs path unavailable (${e.message}) — using the <video> element`);
    }
  }
  return extractElement(file, opts, log, onProgress);
}

/** Quick sniff: is this file a video the pipeline should extract from? */
export function isVideoFile(f) {
  return /^video\//.test(f.type) || /\.(mp4|mov|m4v|webm|mkv)$/i.test(f.name || '');
}
