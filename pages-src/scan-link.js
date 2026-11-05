import { matchSift } from './vendor/splat/sfm/sift.js';
import { ransacE, makeRng } from './vendor/splat/sfm/geometry.js';

// An early connection check, not a replacement for the full solve. It asks
// what the solver asks of neighboring photos: 15 inliers, 40% of the matches.
export function linkedViews(a, b) {
  const pairs = matchSift(a.desc, a.n, b.desc, b.n);
  if (pairs.length < 30) return false; // two entries per match
  const normalized = (f, i) => [(f.x[i] - f.w / 2) / f.focal, (f.y[i] - f.h / 2) / f.focal];
  const x1 = [], x2 = [];
  for (let i = 0; i < pairs.length; i += 2) {
    x1.push(normalized(a, pairs[i])); x2.push(normalized(b, pairs[i + 1]));
  }
  const result = ransacE(x1, x2, 2 / Math.min(a.focal, b.focal), makeRng(71), 200);
  return !!result && result.inliers.length >= 15 && result.inliers.length >= x1.length * 0.4;
}

const RECENT_VIEWS = 3;
const SEED_VIEWS = 4;
// the third miss in a row is kept anyway
const MAX_MISSES = 3;

/**
 * Which capture candidates to keep. A photo that links to a recent view is
 * kept. One that does not is held back, not thrown away: on real scans
 * neighbors link strongly except at jumps (three in the 58 photos of a can:
 * around a counter's corner and at each new step), where the photos after the
 * jump are good and only the way back to the old ones is missing. Refusing
 * them lost 42 of those 58, though the solver placed 57. So a photo that links
 * to a held one brings it back, both starting a new stretch for the solver to
 * join up, and a run of misses ends with a photo kept regardless: a scan
 * stalled on this check is worse than a weak view.
 */
export class LinkGate {
  recent = [];
  seeds = [];
  held = [];
  constructor(link = linkedViews) { this.link = link; }

  /** A view from an existing scan: stays linkable while new photos come in. */
  seed(view) { this.seeds = [...this.seeds, view].slice(-SEED_VIEWS); }

  keep(view, promoted = [], forced = false) {
    this.recent = [view, ...this.recent].slice(0, RECENT_VIEWS);
    this.held = [];
    return { accepted: true, promoted, forced };
  }

  check(view, { newChain = false } = {}) {
    const known = [...this.recent, ...this.seeds];
    if (newChain || !known.length || known.some((other) => this.link(other, view))) return this.keep(view);
    const partner = this.held.find((other) => this.link(other, view));
    if (partner) {
      this.recent = [partner, ...this.recent].slice(0, RECENT_VIEWS);
      return this.keep(view, [partner.name]);
    }
    if (this.held.length + 1 >= MAX_MISSES) return this.keep(view, [], true);
    this.held = [view, ...this.held];
    return { accepted: false, promoted: [], forced: false };
  }
}
