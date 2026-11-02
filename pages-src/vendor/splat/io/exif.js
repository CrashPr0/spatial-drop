// exif.js — the one EXIF fact the camera solver can use: the focal length.
//
// readExifFocal(file) -> { f35, fmm, make, model } | null
//   f35  FocalLengthIn35mmFilm (mm), the 35 mm-equivalent focal — phones and
//        most cameras write it, and it converts to pixels without knowing the
//        sensor: f_px = f35 * diagonal_px / 43.27 (CIPA's definition of the
//        35 mm equivalent is by diagonal, so it holds for 4:3 and 3:2 alike)
//   fmm  FocalLength (mm), the physical focal — only usable with a known sensor
//
// Reads the file header only (JPEG: the APP1 segment; HEIF/HEIC: the meta
// box's Exif item, fetched by its iloc offset). Everything unknown -> null;
// this must never make a decode fail.

const te = new TextDecoder('latin1');

async function bytes(file, start, len) {
  const blob = file.slice(start, start + len);
  return new Uint8Array(await blob.arrayBuffer());
}

/** Parse a TIFF/EXIF block (starting at the byte-order mark). */
function parseTiff(u8, base = 0) {
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  if (u8.length < base + 8) return null;
  const le = u8[base] === 0x49 && u8[base + 1] === 0x49;
  if (!le && !(u8[base] === 0x4D && u8[base + 1] === 0x4D)) return null;
  const u16 = (o) => dv.getUint16(o, le), u32 = (o) => dv.getUint32(o, le);
  if (u16(base + 2) !== 42) return null;
  const out = {};
  const readIfd = (off, tags) => {
    if (off + 2 > u8.length) return;
    const n = u16(off);
    for (let i = 0; i < n; i++) {
      const e = off + 2 + i * 12;
      if (e + 12 > u8.length) return;
      const tag = u16(e), type = u16(e + 2), count = u32(e + 4);
      const size = [0, 1, 1, 2, 4, 8, 1, 1, 2, 4, 8, 4, 8][type] || 1;
      const total = size * count;
      const vo = total <= 4 ? e + 8 : base + u32(e + 8);
      if (vo + total > u8.length) continue;
      const value = () => {
        if (type === 3) return u16(vo);
        if (type === 4) return u32(vo);
        if (type === 5) { const num = u32(vo), den = u32(vo + 4); return den ? num / den : 0; }
        if (type === 2) return te.decode(u8.subarray(vo, vo + Math.max(0, count - 1))).replace(/\0.*$/, '').trim();
        return null;
      };
      const h = tags[tag];
      if (h) h(value(), vo);
    }
  };
  const ifd0 = base + u32(base + 4);
  let exifOff = 0;
  readIfd(ifd0, {
    0x010F: (v) => { out.make = v; },
    0x0110: (v) => { out.model = v; },
    0x8769: (v) => { exifOff = v; },
  });
  if (exifOff) {
    readIfd(base + exifOff, {
      0x920A: (v) => { if (v > 0) out.fmm = v; },
      0xA405: (v) => { if (v > 0) out.f35 = v; },
      0xA002: (v) => { out.pixW = v; },
      0xA003: (v) => { out.pixH = v; },
    });
  }
  return out;
}

/** JPEG: walk the marker segments to APP1 'Exif\0\0'. */
async function jpegExif(file) {
  const head = await bytes(file, 0, Math.min(file.size, 256 * 1024));
  if (head[0] !== 0xFF || head[1] !== 0xD8) return null;
  let i = 2;
  while (i + 4 <= head.length) {
    if (head[i] !== 0xFF) return null;
    const marker = head[i + 1];
    if (marker === 0xD8 || (marker >= 0xD0 && marker <= 0xD7) || marker === 0x01) { i += 2; continue; }
    if (marker === 0xDA || marker === 0xD9) return null;   // image data: no APP1 before it
    const len = (head[i + 2] << 8) | head[i + 3];
    if (marker === 0xE1 && te.decode(head.subarray(i + 4, i + 10)) === 'Exif\0\0') {
      let seg = head.subarray(i + 10, i + 2 + len);
      if (i + 2 + len > head.length) seg = (await bytes(file, i + 10, len - 8));
      return parseTiff(seg, 0);
    }
    i += 2 + len;
  }
  return null;
}

/** HEIF/HEIC (ISOBMFF): meta -> iinf (item of type 'Exif') -> iloc (offset,
 *  length); the item payload is a 4-byte offset to the TIFF header. */
async function heifExif(file) {
  const head = await bytes(file, 0, Math.min(file.size, 512 * 1024));
  const dv = new DataView(head.buffer, head.byteOffset, head.byteLength);
  const u32 = (o) => dv.getUint32(o);
  const u16 = (o) => dv.getUint16(o);
  if (head.length < 12 || te.decode(head.subarray(4, 8)) !== 'ftyp') return null;
  // find the top-level 'meta' box
  let p = 0, meta = null;
  while (p + 8 <= head.length) {
    let size = u32(p); const type = te.decode(head.subarray(p + 4, p + 8));
    let hdr = 8;
    if (size === 1) { size = Number(dv.getBigUint64(p + 8)); hdr = 16; }
    if (size === 0) size = head.length - p;
    if (type === 'meta') { meta = { start: p + hdr + 4, end: Math.min(p + size, head.length) }; break; }   // +4: full box version/flags
    if (size < 8) return null;
    p += size;
  }
  if (!meta) return null;
  let exifId = -1;
  const iloc = [];
  // children of meta
  for (let q = meta.start; q + 8 <= meta.end;) {
    const size = u32(q); const type = te.decode(head.subarray(q + 4, q + 8));
    if (size < 8) break;
    if (type === 'iinf') {
      const ver = head[q + 8];
      let r = q + 12; const count = ver === 0 ? u16(r) : u32(r); r += ver === 0 ? 2 : 4;
      for (let k = 0; k < count && r + 8 <= meta.end; k++) {
        const isz = u32(r); const t = te.decode(head.subarray(r + 4, r + 8));
        if (t === 'infe') {
          const v = head[r + 8];
          const id = v >= 2 ? (v === 2 ? u16(r + 12) : u32(r + 12)) : u16(r + 12);
          const itype = te.decode(head.subarray(r + (v === 2 ? 16 : 18), r + (v === 2 ? 20 : 22)));
          if (itype === 'Exif') exifId = id;
        }
        if (isz < 8) break;
        r += isz;
      }
    } else if (type === 'iloc') {
      const ver = head[q + 8];
      const b1 = head[q + 12], b2 = head[q + 13];
      const offSize = b1 >> 4, lenSize = b1 & 15, baseSize = b2 >> 4, idxSize = ver >= 1 ? (b2 & 15) : 0;
      let r = q + 14;
      const count = ver < 2 ? u16(r) : u32(r); r += ver < 2 ? 2 : 4;
      const rd = (sz) => { let v = 0; for (let b = 0; b < sz; b++) v = v * 256 + head[r++]; return v; };
      for (let k = 0; k < count && r < meta.end; k++) {
        const id = ver < 2 ? u16(r) : u32(r); r += ver < 2 ? 2 : 4;
        if (ver >= 1) r += 2;   // construction method
        r += 2;                 // data reference index
        const base = rd(baseSize);
        const ext = u16(r); r += 2;
        for (let x = 0; x < ext; x++) {
          if (idxSize) rd(idxSize);
          const off = rd(offSize), len = rd(lenSize);
          if (x === 0) iloc.push({ id, off: base + off, len });
        }
      }
    }
    q += size;
  }
  if (exifId < 0) return null;
  const loc = iloc.find((l) => l.id === exifId);
  if (!loc || loc.len < 12 || loc.len > 4 * 1024 * 1024) return null;
  const item = await bytes(file, loc.off, loc.len);
  const tiffOff = new DataView(item.buffer, item.byteOffset, item.byteLength).getUint32(0);   // exif_tiff_header_offset
  const start = 4 + tiffOff;
  if (start + 8 > item.length) return null;
  return parseTiff(item.subarray(start), 0);
}

/** The focal facts of one photo file, or null when the file carries none. */
export async function readExifFocal(file) {
  try {
    if (!(file instanceof Blob) || file.size < 64) return null;
    const sig = await bytes(file, 0, 12);
    let r = null;
    if (sig[0] === 0xFF && sig[1] === 0xD8) r = await jpegExif(file);
    else if (te.decode(sig.subarray(4, 8)) === 'ftyp') r = await heifExif(file);
    if (!r || (!r.f35 && !r.fmm)) return null;
    return r;
  } catch { return null; }
}

/** Focal in pixels for an image of w x h pixels from its 35 mm-equivalent
 *  focal (diagonal definition). */
export const focalPxFrom35 = (f35, w, h) => f35 * Math.hypot(w, h) / 43.27;
