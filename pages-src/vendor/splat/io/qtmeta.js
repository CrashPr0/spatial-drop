// qtmeta.js — the static metadata a phone writes into a QuickTime/MP4 container:
// lens model, 35 mm-equivalent focal length and f-number ('mdta' keyed items in
// moov/meta: a 'keys' atom listing the names, an 'ilst' atom holding the values).
// An iPhone 12 Pro Max ultra-wide clip carries lens_model
// "iPhone 12 Pro Max back camera 1.54mm f/2.4", focal_length.35mm_equivalent "14",
// lens_irisfnumber "F2.40" (2026-09-15). The moov usually sits at the file's start
// (a few hundred KB at most) — the last 4 MB are scanned too for moov-at-end files.

const KEYS = ['camera.lens_model', 'camera.focal_length.35mm_equivalent', 'camera.lens_irisfnumber', 'make', 'model'];

function u32(b, p) { return ((b[p] << 24) | (b[p + 1] << 16) | (b[p + 2] << 8) | b[p + 3]) >>> 0; }
function find(b, needle, from = 0) {
  outer: for (let i = from; i <= b.length - needle.length; i++) {
    for (let k = 0; k < needle.length; k++) if (b[i + k] !== needle[k]) continue outer;
    return i;
  }
  return -1;
}
const ascii = (s) => Uint8Array.from(s, (c) => c.charCodeAt(0));
const dec = new TextDecoder();

/** @param {Uint8Array} b  @returns {Record<string, string|number>} */
export function parseQuickTimeMeta(b) {
  const out = {};
  const KEYS_ATOM = ascii('keys'), ILST = ascii('ilst'), MDTA = ascii('mdta'), DATA = ascii('data');
  let pos = 0;
  while (pos < b.length) {
    const i = find(b, KEYS_ATOM, pos); if (i < 0) break; pos = i + 4;
    const count = u32(b, i + 8); if (count < 1 || count > 64) continue;
    const names = []; let p = i + 12; let ok = true;
    for (let k = 0; k < count; k++) {
      const sz = u32(b, p); if (sz < 8 || sz > 300 || p + sz > b.length) { ok = false; break; }
      if (find(b.subarray(p + 4, p + 8), MDTA) !== 0) { ok = false; break; }
      names.push(dec.decode(b.subarray(p + 8, p + sz))); p += sz;
    }
    if (!ok || !names.some((n) => n.startsWith('com.apple.quicktime.'))) continue;
    // the ilst right after the keys atom: items [size][index][data boxes]
    const il = find(b, ILST, p); if (il < 0 || il - p > 64) continue;
    const ilSize = u32(b, il - 4); let q = il + 4; const end = il - 4 + ilSize;
    while (q + 8 <= end && q + 8 <= b.length) {
      const isz = u32(b, q), idx = u32(b, q + 4); if (isz < 8) break;
      let r = q + 8;
      while (r + 16 <= q + isz) {
        const dsz = u32(b, r); if (dsz < 16) break;
        if (find(b.subarray(r + 4, r + 8), DATA) === 0) {
          const type = u32(b, r + 8); const pl = b.subarray(r + 16, r + dsz); const name = (names[idx - 1] || '').replace('com.apple.quicktime.', '');
          if (KEYS.includes(name)) {
            if (type === 1) out[name] = dec.decode(pl);
            else if (type === 21 || type === 22) out[name] = pl.length === 4 ? (u32(pl, 0) | 0) : pl.length === 2 ? ((pl[0] << 8) | pl[1]) : pl.length === 1 ? pl[0] : null;
            else if (type === 23 && pl.length === 4) out[name] = new DataView(pl.buffer, pl.byteOffset, 4).getFloat32(0);
          }
        }
        r += dsz;
      }
      q += isz;
    }
  }
  return out;
}

/** The container metadata of a video file: { lensModel?, focal35?, fNumber?, make?, model? }. */
export async function readVideoMeta(file) {
  try {
    const head = new Uint8Array(await file.slice(0, Math.min(file.size, 4 << 20)).arrayBuffer());
    let m = parseQuickTimeMeta(head);
    if (!('camera.focal_length.35mm_equivalent' in m) && file.size > (4 << 20)) {
      const tail = new Uint8Array(await file.slice(file.size - (4 << 20)).arrayBuffer());
      m = { ...parseQuickTimeMeta(tail), ...m };
    }
    const f35 = parseFloat(m['camera.focal_length.35mm_equivalent']);
    const fn = parseFloat(String(m['camera.lens_irisfnumber'] || '').replace(/^F/i, ''));
    return {
      lensModel: m['camera.lens_model'] || null,
      focal35: Number.isFinite(f35) && f35 > 0 ? f35 : null,
      fNumber: Number.isFinite(fn) ? fn : null,
      make: m.make || null, model: m.model || null,
    };
  } catch { return { lensModel: null, focal35: null, fNumber: null, make: null, model: null }; }
}
