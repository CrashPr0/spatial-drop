// Live capture guidance. The camera solve chains photos together through the
// features they share, so each new photo has to overlap the one before it.
// Matching a tiny copy of the live frame against the last kept photo tells
// the person whether they have moved far enough, too far, or lost it.

export interface FrameMatch {
  /** median shift of the last photo's tiles found in the live frame, as a fraction of the frame */
  dx: number;
  dy: number;
  /** share of the last photo's textured tiles still found, 0–1 */
  overlap: number;
  /** median zero-mean normalized cross-correlation of the found tiles, −1–1 */
  score: number;
}

export type CaptureAdvice = "first" | "move" | "hold" | "lost";

/** Box-filter a grayscale image down by an integer factor. */
export function shrink(gray: Float32Array, width: number, height: number, factor: number) {
  const w = Math.floor(width / factor), h = Math.floor(height / factor);
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let sum = 0;
      for (let v = 0; v < factor; v++) {
        const row = (y * factor + v) * width + x * factor;
        for (let u = 0; u < factor; u++) sum += gray[row + u];
      }
      out[y * w + x] = sum / (factor * factor);
    }
  }
  return { gray: out, width: w, height: h };
}

/**
 * How much of the last photo the live frame still shows. Orbiting an object
 * keeps the object centered while the background swings past it, so no
 * single shift lines the frames up; instead each textured tile of the last
 * photo is searched for on its own, like a tiny feature matcher.
 * `overlap` is the share of those tiles still found, `dx`/`dy` their median
 * shift, which says which way the view is travelling.
 */
export function matchFrames(previous: Float32Array, current: Float32Array, width: number, height: number, { tile = 16, reach = 0.3, minScore = 0.8 } = {}): FrameMatch {
  const maxX = Math.round(width * reach), maxY = Math.round(height * reach);
  const shiftsX: number[] = [], shiftsY: number[] = [], scores: number[] = [];
  let textured = 0;
  for (let ty = 0; ty + tile <= height; ty += tile) {
    for (let tx = 0; tx + tile <= width; tx += tile) {
      let sa = 0, saa = 0;
      for (let y = 0; y < tile; y++) for (let x = 0; x < tile; x++) {
        const v = previous[(ty + y) * width + tx + x];
        sa += v; saa += v * v;
      }
      const n = tile * tile;
      const varA = saa - (sa * sa) / n;
      // flat tiles (a blank wall) match anywhere: leave them out
      if (varA < n * 25) continue;
      textured++;
      let best = -1, bestX = 0, bestY = 0;
      for (let dy = -maxY; dy <= maxY; dy++) {
        if (ty + dy < 0 || ty + dy + tile > height) continue;
        for (let dx = -maxX; dx <= maxX; dx++) {
          if (tx + dx < 0 || tx + dx + tile > width) continue;
          let sb = 0, sbb = 0, sab = 0;
          for (let y = 0; y < tile; y++) {
            let a = (ty + y) * width + tx, b = (ty + y + dy) * width + tx + dx;
            for (let x = 0; x < tile; x++, a++, b++) {
              const vb = current[b];
              sb += vb; sbb += vb * vb; sab += previous[a] * vb;
            }
          }
          const varB = sbb - (sb * sb) / n;
          if (varB < 1e-6) continue;
          const score = (sab - (sa * sb) / n) / Math.sqrt(varA * varB);
          // ties go to the smaller shift
          if (score > best + 1e-3 || (score > best - 1e-3 && Math.abs(dx) + Math.abs(dy) < Math.abs(bestX) + Math.abs(bestY))) {
            best = score; bestX = dx; bestY = dy;
          }
        }
      }
      if (best >= minScore) { shiftsX.push(bestX); shiftsY.push(bestY); scores.push(best); }
    }
  }
  const median = (values: number[]) => values.length ? [...values].sort((a, b) => a - b)[values.length >> 1] : 0;
  return {
    dx: median(shiftsX) / width,
    dy: median(shiftsY) / height,
    overlap: textured ? scores.length / textured : 0,
    score: median(scores),
  };
}

/**
 * What the person should do next. On an orbit at arm's length, `keepBelow`
 * is reached after about 10° of travel and `lostBelow` past about 18°:
 * enough new view to add something, enough shared view for the solve to
 * link the photo to the one before.
 */
export function adviseCapture(match: FrameMatch | null, { keepBelow = 0.35, lostBelow = 0.1 } = {}): CaptureAdvice {
  if (!match) return "first";
  if (match.overlap < lostBelow) return "lost";
  return match.overlap < keepBelow ? "hold" : "move";
}

export type Heading = -1 | 0 | 1;
/** left/right: around the object; up/down: raise or lower the phone */
export type Arrow = "left" | "right" | "up" | "down" | "either" | null;

/**
 * Which way to point. `heading` is the way the orbit has been going (+1:
 * the view travels right), learned from the photos kept so far. Lost views
 * point back toward the last photo.
 */
export function pointTo(advice: CaptureAdvice, match: FrameMatch | null, heading: Heading, width: number): Arrow {
  const side = (sign: number): Arrow => (sign > 0 ? "right" : "left");
  if (advice === "move") return heading ? side(heading) : "either";
  if (advice !== "lost") return null;
  // the tiles still found have slid one way: go back the other
  if (match && Math.abs(match.dx * width) >= 2) return side(-match.dx);
  return heading ? side(-heading) : null;
}
