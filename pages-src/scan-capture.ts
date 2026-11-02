// Live capture guidance. The camera solve chains photos together through the
// features they share, so each new photo has to overlap the one before it.
// Matching a tiny copy of the live frame against the last kept photo tells
// the person whether they have moved far enough, too far, or lost it.

export interface FrameMatch {
  /** where the last photo's content sits in the live frame, as a fraction of the frame */
  dx: number;
  dy: number;
  /** share of the live frame still covered by the last photo, 0–1 */
  overlap: number;
  /** zero-mean normalized cross-correlation at that shift, −1–1 */
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
 * Best translation of `previous` inside `current` by exhaustive search over
 * shifts that keep at least `minOverlap` of the frame in common. Orbiting an
 * object mostly slides the view sideways, so a shift is a fair stand-in for
 * how much two photos share.
 */
export function matchFrames(previous: Float32Array, current: Float32Array, width: number, height: number, minOverlap = 0.2): FrameMatch {
  let best: FrameMatch = { dx: 0, dy: 0, overlap: 0, score: -1 };
  let bestRank = -Infinity;
  const maxX = Math.floor(width * (1 - minOverlap)), maxY = Math.floor(height * 0.5);
  for (let dy = -maxY; dy <= maxY; dy++) {
    for (let dx = -maxX; dx <= maxX; dx++) {
      const w = width - Math.abs(dx), h = height - Math.abs(dy);
      if (w * h < minOverlap * width * height) continue;
      // previous (x, y) lands on current (x + dx, y + dy)
      const px = Math.max(0, -dx), py = Math.max(0, -dy);
      let sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
      for (let y = 0; y < h; y++) {
        let a = (py + y) * width + px, b = (py + y + dy) * width + px + dx;
        for (let x = 0; x < w; x++, a++, b++) {
          const va = previous[a], vb = current[b];
          sa += va; sb += vb; saa += va * va; sbb += vb * vb; sab += va * vb;
        }
      }
      const n = w * h;
      const cov = sab - (sa * sb) / n;
      const varA = saa - (sa * sa) / n, varB = sbb - (sb * sb) / n;
      const score = varA > 1e-6 && varB > 1e-6 ? cov / Math.sqrt(varA * varB) : -1;
      // a slight pull toward small shifts: big shifts compare less of the
      // frame and win by chance more often
      const rank = score - 0.05 * (1 - n / (width * height));
      if (rank > bestRank) {
        bestRank = rank;
        best = { dx: dx / width, dy: dy / height, overlap: n / (width * height), score };
      }
    }
  }
  return best;
}

/**
 * What the person should do next. Photos are kept between `keepBelow` and
 * `lostBelow` overlap: enough new view to add something, enough shared view
 * for the solve to link it to the last photo.
 */
export function adviseCapture(match: FrameMatch | null, { keepBelow = 0.8, lostBelow = 0.4, minScore = 0.45 } = {}): CaptureAdvice {
  if (!match) return "first";
  if (match.score < minScore || match.overlap < lostBelow) return "lost";
  return match.overlap < keepBelow ? "hold" : "move";
}
