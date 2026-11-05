// One candidate at a time, and the few views LinkGate holds, bound capture-time memory.
import { decodeFrames } from './vendor/splat/io/frames.js';
import { detectSift } from './vendor/splat/sfm/sift.js';
import { LinkGate } from './scan-link.js';
const gate = new LinkGate();
self.onmessage = async ({ data: { id, file, referenceOnly, newChain } }) => {
  try {
    const [frame] = await decodeFrames([file], { trainMaxDim: 2, exif: false });
    if (!frame) throw new Error('Photo could not be decoded');
    frame.thumb.width = frame.thumb.height = 0;
    const features = detectSift(frame.gray, frame.fw, frame.fh, 3900, 0, 1);
    const n = Math.min(1000, features.n);
    const view = { name: file.name, n, desc: features.desc.slice(0, n * 128), x: features.x.slice(0, n), y: features.y.slice(0, n),
      w: frame.fw, h: frame.fh, focal: 26 * Math.hypot(frame.fw, frame.fh) / 43.27 };
    if (referenceOnly) {
      gate.seed(view);
      self.postMessage({ id, accepted: true });
      return;
    }
    self.postMessage({ id, ...gate.check(view, { newChain }), fw: frame.fw, fh: frame.fh, features },
      [features.x.buffer, features.y.buffer, features.scale.buffer, features.angle.buffer, features.desc.buffer]);
  } catch {
    self.postMessage({ id, accepted: null }); // Unknown: preserve the photo for the full solver.
  }
};
