// Orbit coverage, Polycam-style: a ring of segments around the object that
// fill in as each side gets photographed. Where a photo sits on the ring comes
// from the camera's real orientation (the phone's sensors, which see the
// object at the center of the view); when a phone has none, from how far the
// view slid against the last photo. A stretch between two photos only counts
// once the overlap check says they share enough view for the camera solve to
// link them, so a fast swing past the object leaves a hole, not a fill.

export const SEGMENTS = 12;
const SEGMENT_DEG = 360 / SEGMENTS;
/** a photo sees the object from this far either side of where it was taken */
export const PHOTO_REACH_DEG = 12;
/** share of the last photo still visible for two photos to count as linked */
export const LINK_OVERLAP = 0.1;
/** a sensor jump bigger than this between linked photos is drift, not travel */
export const MAX_LINK_DEG = 45;
/** a segment counts as scanned from this fill on */
export const SCANNED_FILL = 0.8;
/** rough orbit travel for a fully new view, when only the image is available */
const VISUAL_DEG_PER_LOSS = 35;

export interface Orientation { alpha: number; beta: number; gamma: number }

const rad = Math.PI / 180;
export const wrap = (deg: number) => ((deg % 360) + 360) % 360;
/** signed shortest turn from a to b, in (−180, 180] */
export const turn = (a: number, b: number) => {
  const d = wrap(b - a);
  return d > 180 ? d - 360 : d;
};

/**
 * Compass heading and elevation of the back camera's viewing direction from
 * a DeviceOrientation reading (W3C spec formulas for Z-X'-Y'' angles).
 */
export function viewDirection({ alpha, beta, gamma }: Orientation) {
  const cA = Math.cos(alpha * rad), sA = Math.sin(alpha * rad);
  const cB = Math.cos(beta * rad), sB = Math.sin(beta * rad);
  const cG = Math.cos(gamma * rad), sG = Math.sin(gamma * rad);
  const east = -cA * sG - sA * sB * cG;
  const north = -sA * sG + cA * sB * cG;
  return { heading: wrap(Math.atan2(east, north) / rad), elevation: Math.asin(Math.max(-1, Math.min(1, -cB * cG))) / rad };
}

export interface Coverage {
  /** one flag per degree of the ring the photos have seen */
  seen: Uint8Array;
  /** where the last photo sits, degrees clockwise from where the scan began; null before the first */
  last: number | null;
  /** photos kept that were not linked to the one before (a restart of the chain) */
  breaks: number;
}

export const createCoverage = (): Coverage => ({ seen: new Uint8Array(360), last: null, breaks: 0 });

function mark(seen: Uint8Array, from: number, span: number) {
  for (let i = 0; i <= span; i++) seen[Math.floor(wrap(from + i))] = 1;
}

/**
 * Record a kept photo at ring angle `at`. `overlap` is how much of the
 * previous photo it still shows. Linked photos fill the arc between them;
 * an unlinked one fills only its own reach and starts a new chain.
 */
export function addPhotoToCoverage(coverage: Coverage, at: number, overlap: number) {
  at = wrap(at);
  mark(coverage.seen, at - PHOTO_REACH_DEG, 2 * PHOTO_REACH_DEG);
  if (coverage.last !== null) {
    const step = turn(coverage.last, at);
    if (overlap >= LINK_OVERLAP && Math.abs(step) <= MAX_LINK_DEG) {
      mark(coverage.seen, step >= 0 ? coverage.last : at, Math.abs(step));
    } else coverage.breaks++;
  }
  coverage.last = at;
}

/** 0–1 fill of each segment; segment 0 starts at the scan's first photo, going clockwise */
export function segmentFill(coverage: Coverage) {
  // segments are centred on multiples of 30°, the first photo in the middle of segment 0
  return Array.from({ length: SEGMENTS }, (_, i) => {
    let count = 0;
    for (let d = 0; d < SEGMENT_DEG; d++) count += coverage.seen[Math.floor(wrap(i * SEGMENT_DEG - SEGMENT_DEG / 2 + d))];
    return count / SEGMENT_DEG;
  });
}

export const coverageFraction = (coverage: Coverage) => segmentFill(coverage).reduce((sum, fill) => sum + Math.min(1, fill / SCANNED_FILL), 0) / SEGMENTS;

/**
 * The nearest unscanned segment from `at`, and which way round to go for it.
 * `direction` is the ring direction: "cw" raises the angle, which is the
 * camera walking to the left. `prefer` breaks near-ties. Null when the ring
 * is complete.
 */
export function nearestGap(coverage: Coverage, at: number, prefer?: "cw" | "ccw") {
  const fill = segmentFill(coverage);
  let best: { segment: number; direction: "cw" | "ccw"; degrees: number } | null = null;
  for (let i = 0; i < SEGMENTS; i++) {
    if (fill[i] >= SCANNED_FILL) continue;
    const d = turn(at, i * SEGMENT_DEG);
    // near-ties go the way the orbit is already travelling
    const degrees = Math.abs(d) - (prefer && (d >= 0) === (prefer === "cw") ? 5 : 0);
    if (!best || degrees < best.degrees) best = { segment: i, direction: d >= 0 ? "cw" : "ccw", degrees };
  }
  return best;
}

/** Names the open sides in words, as "left, behind", relative to where the scan began. */
export function missingSides(coverage: Coverage) {
  const names = ["front", "front left", "left", "back left", "back", "back right", "right", "front right"];
  const fill = segmentFill(coverage);
  const open = new Set<string>();
  for (let i = 0; i < SEGMENTS; i++) {
    if (fill[i] >= SCANNED_FILL) continue;
    // ring angle clockwise from above, with the start at the front
    open.add(names[Math.round(((i * SEGMENT_DEG) / 360) * names.length) % names.length]);
  }
  return [...open];
}

/** SVG path of one ring segment (a donut slice), angles clockwise from the bottom of the ring. */
export function segmentPath(index: number, outer = 44, inner = 30, gap = 2.5) {
  const a0 = (index * SEGMENT_DEG - SEGMENT_DEG / 2) + gap, a1 = (index * SEGMENT_DEG + SEGMENT_DEG / 2) - gap;
  // 0° is the bottom (where the person starts), clockwise on screen
  const point = (r: number, deg: number) => `${(50 - r * Math.sin(deg * rad)).toFixed(2)} ${(50 + r * Math.cos(deg * rad)).toFixed(2)}`;
  return `M${point(outer, a0)}A${outer} ${outer} 0 0 0 ${point(outer, a1)}L${point(inner, a1)}A${inner} ${inner} 0 0 1 ${point(inner, a0)}Z`;
}

/** Position of a ring angle as SVG coordinates on the circle of the given radius. */
export function ringPoint(deg: number, radius: number) {
  return { x: 50 - radius * Math.sin(deg * rad), y: 50 + radius * Math.cos(deg * rad) };
}

/**
 * Tracks where the camera is on the ring. Sensor readings give the ring angle
 * directly: orbiting the object keeps it centred, so the view heading turns
 * one-for-one with the camera's position around it. Without them the angle
 * is dead-reckoned from the visual shift, which is coarse but still keeps
 * the ring honest about direction and rough extent.
 */
export class PoseTracker {
  private origin: number | null = null;
  private reading: { heading: number; elevation: number } | null = null;

  get hasSensor() { return this.reading !== null; }

  /** Feed a DeviceOrientation reading. */
  sense(orientation: Orientation) {
    this.reading = viewDirection(orientation);
  }

  /** Pin the ring's zero to where the camera is looking now. */
  anchor() {
    this.origin = this.reading?.heading ?? null;
  }

  /**
   * Ring angle of the camera now. It follows the compass heading of the view,
   * so walking to the right lowers it: on the ring, drawn from above with the
   * start at the bottom, that is toward the right of the screen.
   */
  angle(): number | null {
    if (this.reading && this.origin !== null) return wrap(turn(this.origin, this.reading.heading));
    return this.reading ? 0 : null;
  }

  /**
   * Ring angle for a photo kept now. `shift` is the horizontal slide of the
   * last photo's tiles in the live frame (fraction of width, + right) and
   * `overlap` how much of it is still seen; used only when there is no sensor.
   */
  angleForKeep(last: number | null, shift: number, overlap: number) {
    const sensed = this.angle();
    if (sensed !== null) return sensed;
    if (last === null) return 0;
    // tiles sliding right means the camera went right, which lowers the angle
    const sign = shift > 0 ? -1 : 1;
    return wrap(last + sign * Math.max(6, (1 - overlap) * VISUAL_DEG_PER_LOSS));
  }
}
