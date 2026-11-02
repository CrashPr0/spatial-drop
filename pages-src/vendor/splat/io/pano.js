// pano.js — 360 equirectangular input, sliced into a cubemap camera rig.
//
// A pano is not one camera the pinhole pipeline can use, but it IS six: the
// slicer resamples the sphere into six overlapping pinhole faces around a
// shared centre. The faces train as ordinary pinhole targets; the solver
// gets told they form a rig ({ id, R } per face — one 6-DOF pose for all
// six, see sfm opts.rigs) and that their focal is known exactly (a sliced
// face's focal is size/2 / tan(fov/2) by construction, not a camera trait).

/** Rig->face rotations for the six cube faces, in the camera convention
 *  (x right, y DOWN, z forward): +z, +x, -z, -x, up (-y), down (+y). */
export const FACE_ROTS = [
  [1, 0, 0, 0, 1, 0, 0, 0, 1],
  [0, 0, -1, 0, 1, 0, 1, 0, 0],
  [-1, 0, 0, 0, 1, 0, 0, 0, -1],
  [0, 0, 1, 0, 1, 0, -1, 0, 0],
  [1, 0, 0, 0, 0, -1, 0, 1, 0],
  [1, 0, 0, 0, 0, 1, 0, -1, 0],
];

/** An equirectangular pano is a 2:1 image (within tolerance — some cameras
 *  write 1.99 or 2.01 after stitching). */
export function isEquirect(w, h) {
  return h > 0 && Math.abs(w / h - 2) < 0.05;
}

/** Image dimensions from the file HEADER — no decode. JPEG (SOF scan) and
 *  PNG (IHDR) cover what 360 cameras write; anything else returns null and
 *  is simply not treated as a pano. Canvas-likes carry width/height. */
export async function probeImageSize(src) {
  if (src && typeof src.width === 'number' && typeof src.height === 'number') {
    return { w: src.width, h: src.height };
  }
  if (!(src instanceof Blob)) return null;
  // 2MB window: camera JPEGs (Sony, iPhone) park multi-hundred-KB EXIF
  // preview blobs before the SOF marker — a 256KB slice missed them and
  // silently disabled decode-to-target downstream
  const head = new Uint8Array(await src.slice(0, 2097152).arrayBuffer());
  // PNG: 8-byte signature, IHDR width/height at 16/20
  if (head.length > 24 && head[0] === 0x89 && head[1] === 0x50) {
    const dv = new DataView(head.buffer);
    return { w: dv.getUint32(16), h: dv.getUint32(20) };
  }
  // JPEG: walk the TOP-LEVEL segment stream to the first SOFn. STRICT: no
  // byte-resync — a resyncing walker wanders into EXIF-embedded preview
  // JPEGs (Sony, iPhone) and reports the thumbnail's dimensions. Any
  // nonconforming stream returns null (callers fall back to a full decode).
  if (head.length > 4 && head[0] === 0xff && head[1] === 0xd8) {
    let p = 2;
    while (p + 9 < head.length) {
      if (head[p] !== 0xff) return null;      // desynced: distrust everything
      let q = p;
      while (head[q + 1] === 0xff && q + 9 < head.length) q++; // fill bytes
      const m = head[q + 1];
      if (m === 0x01 || (m >= 0xd0 && m <= 0xd7)) { p = q + 2; continue; }
      if (m === 0xd9 || m === 0xda) return null; // EOI/SOS before any SOF
      const len = (head[q + 2] << 8) | head[q + 3];
      if (len < 2) return null;
      if (m >= 0xc0 && m <= 0xcf && m !== 0xc4 && m !== 0xc8 && m !== 0xcc) {
        return { w: (head[q + 7] << 8) | head[q + 8], h: (head[q + 5] << 8) | head[q + 6] };
      }
      p = q + 2 + len;
    }
  }
  return null;
}

/** Face size for a given pano width: a 90-degree face covers a quarter of
 *  the pano's width — going beyond that invents pixels. Clamped to keep
 *  feature extraction and memory sane. */
export function faceSizeFor(panoW) {
  return Math.max(512, Math.min(1280, Math.round(panoW / 4 / 64) * 64));
}

/**
 * Slice one equirectangular image into six overlapping pinhole faces.
 * @param {ImageBitmap|HTMLCanvasElement} source  the decoded pano
 * @param {number} size    square face size in pixels
 * @param {number} fovDeg  face field of view (>90 gives adjacent overlap)
 * @returns {{ faces: OffscreenCanvas[], f: number, size: number }}
 */
export function sliceEquirect(source, size, fovDeg = 100) {
  const W = source.width, H = source.height;
  const rd = new OffscreenCanvas(W, H).getContext('2d', { willReadFrequently: true });
  rd.drawImage(source, 0, 0);
  const src = rd.getImageData(0, 0, W, H).data;

  const f = (size / 2) / Math.tan((fovDeg / 2) * Math.PI / 180);
  const c = size / 2;
  const faces = [];
  for (let k = 0; k < 6; k++) {
    const Rf = FACE_ROTS[k];
    const cv = new OffscreenCanvas(size, size);
    const ctx = cv.getContext('2d');
    const out = ctx.createImageData(size, size);
    const od = out.data;
    for (let py = 0; py < size; py++) {
      const vy = (py + 0.5 - c) / f;
      for (let px = 0; px < size; px++) {
        const vx = (px + 0.5 - c) / f;
        // face ray -> rig frame (Rf is rig->face; apply the transpose)
        const dx = Rf[0] * vx + Rf[3] * vy + Rf[6];
        const dy = Rf[1] * vx + Rf[4] * vy + Rf[7];
        const dz = Rf[2] * vx + Rf[5] * vy + Rf[8];
        const inv = 1 / Math.hypot(dx, dy, dz);
        const lon = Math.atan2(dx, dz);
        const lat = Math.asin(-dy * inv);
        // exact inverse of the equirect mapping used everywhere else here
        let sx = (lon / (2 * Math.PI) + 0.5) * W - 0.5;
        let sy = (0.5 - lat / Math.PI) * H - 0.5;
        // bilinear, wrapping in longitude, clamped in latitude
        const x0 = Math.floor(sx), y0 = Math.floor(sy);
        const ax = sx - x0, ay = sy - y0;
        const xw0 = ((x0 % W) + W) % W, xw1 = (xw0 + 1) % W;
        const yc0 = Math.min(H - 1, Math.max(0, y0)), yc1 = Math.min(H - 1, Math.max(0, y0 + 1));
        const i00 = (yc0 * W + xw0) * 4, i10 = (yc0 * W + xw1) * 4;
        const i01 = (yc1 * W + xw0) * 4, i11 = (yc1 * W + xw1) * 4;
        const w00 = (1 - ax) * (1 - ay), w10 = ax * (1 - ay), w01 = (1 - ax) * ay, w11 = ax * ay;
        const oi = (py * size + px) * 4;
        od[oi] = src[i00] * w00 + src[i10] * w10 + src[i01] * w01 + src[i11] * w11;
        od[oi + 1] = src[i00 + 1] * w00 + src[i10 + 1] * w10 + src[i01 + 1] * w01 + src[i11 + 1] * w11;
        od[oi + 2] = src[i00 + 2] * w00 + src[i10 + 2] * w10 + src[i01 + 2] * w01 + src[i11 + 2] * w11;
        od[oi + 3] = 255;
      }
    }
    ctx.putImageData(out, 0, 0);
    faces.push(cv);
  }
  // release the full-res working copy NOW — iOS frees canvas stores lazily,
  // and a 6080x3040 pano's read canvas is 74MB of already-dead memory
  rd.canvas.width = 0; rd.canvas.height = 0;
  return { faces, f, size };
}
