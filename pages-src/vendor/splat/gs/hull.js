// hull.js — a visual hull carved from the subject masks, used as a hard
// "nothing may exist here" volume during refinement.
//
// Why a volume and not a loss: supervising rendered opacity (shaders.js covW)
// only governs what GROWS. It cannot remove a Gaussian that has already
// saturated, because dalpha/d(logit opacity) = o(1-o)(...) vanishes as o -> 1
// — a converged opaque splat sits in the sigmoid's dead zone and no opacity
// gradient reaches it again. Density control is the other half of every masked
// -3DGS recipe, and this is the cheapest correct form of it: carve the space
// the silhouettes agree is empty, then let refine() treat anything inside that
// space as dead capacity and relocate it onto the subject.
//
// Carving is conservative on purpose. A voxel only loses a vote from a view
// that sees it AND calls it KNOWN-empty (alpha <= 0.25); the matte's soft edge
// votes neither way, and a voxel outside a view's frame is not evidence of
// anything.

/**
 * @param {object} recon   solved reconstruction (cams + subject points)
 * @param {Array} frames   decoded frames; rgb carries the mask sentinels
 * @param {object} [opts]
 * @param {number} [opts.res=128]     voxels along the longest axis
 * @param {number} [opts.margin=0.15] bbox padding, as a fraction of its size
 * @param {number} [opts.keep=0.85]   min fraction of seeing views that must
 *   NOT call the voxel empty. Below 1 so one bad matte cannot punch a hole.
 * @param {number} [opts.minViews=8]  a voxel seen by fewer views is left alive
 *   (not enough evidence to carve it away)
 * @param {number} [opts.kMad=4]      half-extent of the carving box, in median
 *   absolute deviations of the subject cloud
 * @param {Array<number[]>} [opts.include] points the box must contain (joints, floor)
 * @returns {object|null} hull, or null when the frames carry no masks
 */
export function buildVisualHull(recon, frames, opts = {}) {
  const { res = 128, margin = 0.15, keep = 0.85, minViews = 8, kMad = 4, dilatePx = 0, topY = null, topUp = 1 } = opts;   // topUp: +1 when up is +Y, -1 when the recon's Y points down
  // dilatePx: the subject is grown by this many matte pixels before it votes to carve — a
  // matte that loses fingers, foot edges or hair in a few frames used to carve them out of
  // the hull (10 misses of 65 kill a voxel). A separable max filter over the alpha, per frame,
  // on the fly. topY: voxels above this Y (the scene's up) die regardless of evidence — an
  // eye-level orbit has the space above the head OUTSIDE its frames, and out of frame is
  // "no evidence", which left an uncarved column up to the box's top (ceiling splats on
  // Filip's export, 2026-09-17). The cut stage sets it from the landmarks.
  const dilated = (im, r) => {
    if (!(r > 0)) return im.alpha;
    const W = im.tw, H = im.th, a = im.alpha, tmp = new Uint8Array(W * H), out = new Uint8Array(W * H);
    for (let y = 0; y < H; y++) { const o = y * W; for (let x = 0; x < W; x++) { let m = 0; for (let k = Math.max(0, x - r); k <= Math.min(W - 1, x + r); k++) { const v = a[o + k]; if (v > m) m = v; } tmp[o + x] = m; } }
    for (let x = 0; x < W; x++) { for (let y = 0; y < H; y++) { let m = 0; for (let k = Math.max(0, y - r); k <= Math.min(H - 1, y + r); k++) { const v = tmp[k * W + x]; if (v > m) m = v; } out[y * W + x] = m; } }
    return out;
  };
  if (!frames.some((f) => f.alpha && f.emptyFrac > 0)) return null;
  const pts = recon.points;
  if (!pts || pts.length < 8) return null;

  // Bounds from the mask-filtered cloud — but percentiles are NOT robust
  // enough here. A silhouette test cannot remove a point that lies along the
  // viewing ray THROUGH the subject in every view, so the filtered cloud still
  // has a long tail of far stragglers: on the Lisa orbit the 1-99 percentile
  // box came out 20 x 9 x 24 units, and the person was 0.1% of it. Median
  // +/- k*MAD finds the actual body (0.56 x 1.75 x 0.55 there).
  const med = [0, 0, 0], mad = [0, 0, 0];
  const lo = [0, 0, 0], hi = [0, 0, 0];
  const mid = (arr) => { const a = Float64Array.from(arr).sort(); return a[a.length >> 1]; };
  for (let a = 0; a < 3; a++) {
    const v = Float64Array.from(pts.map((p) => p.X[a])).sort();
    med[a] = v[v.length >> 1];
    mad[a] = mid(pts.map((p) => Math.abs(p.X[a] - med[a]))) || 1e-3;
    // MAD alone collapsed to a 1 cm box on a cloud whose subject points sat in
    // one tight cluster (2026-09-13, the app's 1080p test clip): never let the
    // box be narrower than the cloud's 10-90 percentile span
    const p10 = v[Math.floor(0.1 * (v.length - 1))], p90 = v[Math.floor(0.9 * (v.length - 1))];
    const half = Math.max(kMad * mad[a], 0.5 * (p90 - p10)) * (1 + margin);
    lo[a] = med[a] - half; hi[a] = med[a] + half;
  }
  // opts.include: points the box must contain (the avatar's joints and its floor
  // point). The MAD box is symmetric about the cloud's median, and a person's
  // features sit on the head and torso: the box reached 40 cm above the floor
  // and cut the feet off (2026-09-15). Padded by `margin` of the box's longest side.
  if (Array.isArray(opts.include) && opts.include.length) {
    const pad = margin * Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
    for (const P of opts.include) {
      if (!P || !Number.isFinite(P[0] + P[1] + P[2])) continue;
      for (let a = 0; a < 3; a++) { lo[a] = Math.min(lo[a], P[a] - pad); hi[a] = Math.max(hi[a], P[a] + pad); }
    }
  }
  const span = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
  const cell = span / res;
  const dim = [
    Math.max(1, Math.ceil((hi[0] - lo[0]) / cell)),
    Math.max(1, Math.ceil((hi[1] - lo[1]) / cell)),
    Math.max(1, Math.ceil((hi[2] - lo[2]) / cell)),
  ];
  const nvox = dim[0] * dim[1] * dim[2];
  const alive = new Uint8Array(nvox).fill(1);
  const seen = new Int16Array(nvox);
  const miss = new Int16Array(nvox);
  // Early-out bound. Carving mid-loop on the RATIO is wrong — the counts are
  // partial, so a voxel that draws two empty votes in its first eight views
  // dies even though it would finish at 2/186. (It did: the first build came
  // out 0.1% solid.) This bound is sound instead: once the misses alone exceed
  // what the FULL camera set could ever forgive, no remaining view can save it.
  const killAt = (1 - keep) * recon.cams.length;

  for (const c of recon.cams) {
    const im = frames[c.imgIdx];
    if (!im || !im.alpha || !(im.emptyFrac > 0)) continue;
    const alpha = dilated(im, dilatePx | 0);
    const sx = im.tw / im.fw, sy = im.th / im.fh;
    const { R, t } = c;
    const fy = c.fy ?? c.f;
    let v = 0;
    for (let iz = 0; iz < dim[2]; iz++) {
      const Z = lo[2] + (iz + 0.5) * cell;
      for (let iy = 0; iy < dim[1]; iy++) {
        const Y = lo[1] + (iy + 0.5) * cell;
        for (let ix = 0; ix < dim[0]; ix++, v++) {
          if (!alive[v]) continue;
          const X = lo[0] + (ix + 0.5) * cell;
          const zc = R[6] * X + R[7] * Y + R[8] * Z + t[2];
          if (zc <= 1e-6) continue;
          const u = (c.f * (R[0] * X + R[1] * Y + R[2] * Z + t[0]) / zc + c.cx) * sx;
          const w = (fy * (R[3] * X + R[4] * Y + R[5] * Z + t[1]) / zc + c.cy) * sy;
          if (u < 0 || w < 0 || u >= im.tw || w >= im.th) continue; // not evidence
          seen[v]++;
          // only a KNOWN-empty pixel (alpha <= 0.25) votes to carve; the
          // matte's soft edge abstains, so a fuzzy matte cannot erode the subject
          if (alpha[(w | 0) * im.tw + (u | 0)] <= 64) {
            miss[v]++;
            if (miss[v] > killAt) alive[v] = 0;
          }
        }
      }
    }
  }

  let live = 0, capped = 0;
  for (let i = 0; i < nvox; i++) {
    // a voxel too rarely seen never had enough evidence: leave it alive
    if (alive[i] && seen[i] >= minViews && miss[i] > (1 - keep) * seen[i]) alive[i] = 0;
    if (alive[i] && topY != null) {   // the hard top: nothing above the head, evidence or not
      const iy = ((i / dim[0]) | 0) % dim[1];
      if ((lo[1] + (iy + 0.5) * cell - topY) * topUp > 0) { alive[i] = 0; capped++; }
    }
    if (alive[i]) live++;
  }
  return { lo, cell, dim, alive, nvox, live, fill: live / nvox, capped, dilatePx: dilatePx | 0, topY };
}

/** (x, y, z) -> is this point inside the carved volume? Outside the grid
 *  counts as OUTSIDE: the box was padded around the subject already. */
export function makeHullTest(hull) {
  const { lo, cell, dim, alive } = hull;
  const [dx, dy, dz] = dim;
  return (x, y, z) => {
    const ix = ((x - lo[0]) / cell) | 0;
    if (ix < 0 || ix >= dx) return false;
    const iy = ((y - lo[1]) / cell) | 0;
    if (iy < 0 || iy >= dy) return false;
    const iz = ((z - lo[2]) / cell) | 0;
    if (iz < 0 || iz >= dz) return false;
    return alive[(iz * dy + iy) * dx + ix] === 1;
  };
}

/** (x, y, z, r) -> should this splat be treated as dead capacity?
 *
 *  A point test is not enough: the thing that actually paints a halo is a
 *  splat CENTRED on the subject whose skirt reaches far outside it, and that
 *  passes any centre test. So probe the extent too — six axis samples at r —
 *  and call the splat dead when its centre is outside, or when most of its
 *  body is. Mostly, not entirely: a legitimate surface splat always pokes a
 *  little way out of the hull. */
export function makeSplatTest(hull, opts = {}) {
  const { sigma = 2, maxOut = 4 } = opts;
  const inside = makeHullTest(hull);
  return (x, y, z, rmax) => {
    if (!inside(x, y, z)) return true;
    const r = sigma * rmax;
    if (r < hull.cell) return false;            // smaller than a voxel: centre says it all
    let out = 0;
    if (!inside(x + r, y, z)) out++;
    if (!inside(x - r, y, z)) out++;
    if (!inside(x, y + r, z)) out++;
    if (!inside(x, y - r, z)) out++;
    if (!inside(x, y, z + r)) out++;
    if (!inside(x, y, z - r)) out++;
    return out >= maxOut;
  };
}
