// One pending candidate and three reference views bound capture-time memory.
import { decodeFrames } from './vendor/splat/io/frames.js';
import { detectSift } from './vendor/splat/sfm/sift.js';
import { linkedViews } from './scan-link.js';
let references = [];
self.onmessage = async ({ data: { id, file } }) => {
  try {
    const [frame] = await decodeFrames([file], { trainMaxDim: 2, exif: false });
    if (!frame) throw new Error('Photo could not be decoded');
    frame.thumb.width = frame.thumb.height = 0;
    const features = detectSift(frame.gray, frame.fw, frame.fh, 3900, 0, 1);
    const n = Math.min(1000, features.n);
    const reference = { n, desc: features.desc.slice(0, n * 128), x: features.x.slice(0, n), y: features.y.slice(0, n),
      w: frame.fw, h: frame.fh, focal: 26 * Math.hypot(frame.fw, frame.fh) / 43.27 };
    const accepted = !references.length || references.some((r) => linkedViews(r, reference));
    if (accepted) references = [reference, ...references].slice(0, 3);
    self.postMessage({ id, accepted, fw: frame.fw, fh: frame.fh, features },
      [features.x.buffer, features.y.buffer, features.scale.buffer, features.angle.buffer, features.desc.buffer]);
  } catch {
    self.postMessage({ id, accepted: null }); // Unknown: preserve the photo for the full solver.
  }
};
