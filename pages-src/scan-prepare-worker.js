// One image at a time keeps capture responsive and bounds transient memory.
import { decodeFrames } from './vendor/splat/io/frames.js';
import { detectSift, matchSift } from './vendor/splat/sfm/sift.js';
let previous = null;
self.onmessage = async ({ data: { id, file } }) => {
  try {
    const [frame] = await decodeFrames([file], { trainMaxDim: 2, exif: false });
    if (!frame) throw new Error('Photo could not be decoded');
    frame.thumb.width = frame.thumb.height = 0;
    const f = detectSift(frame.gray, frame.fw, frame.fh, 3900, 0, 1);
    // Bound matching cost while retaining the full extraction for reconstruction.
    const count = Math.min(600, f.n);
    const descriptors = f.desc.slice(0, count * 128);
    const pair = previous ? { previous: previous.name, matches: matchSift(previous.desc, previous.n, descriptors, count).length / 2 } : null;
    previous = { name: file.name, desc: descriptors, n: count };
    self.postMessage({ id, fw: frame.fw, fh: frame.fh, features: f, pair },
      [f.x.buffer, f.y.buffer, f.scale.buffer, f.angle.buffer, f.desc.buffer]);
  } catch {
    previous = null;
    self.postMessage({ id }); // normal reconstruction will retry this image
  }
};
