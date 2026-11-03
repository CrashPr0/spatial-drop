// One image at a time keeps capture responsive and bounds transient memory.
import { decodeFrames } from './vendor/splat/io/frames.js';
import { detectSift } from './vendor/splat/sfm/sift.js';
self.onmessage = async ({ data: { id, file } }) => {
  try {
    const [frame] = await decodeFrames([file], { trainMaxDim: 2, exif: false });
    if (!frame) throw new Error('Photo could not be decoded');
    frame.thumb.width = frame.thumb.height = 0;
    const f = detectSift(frame.gray, frame.fw, frame.fh, 3900, 0, 1);
    self.postMessage({ id, fw: frame.fw, fh: frame.fh, features: f },
      [f.x.buffer, f.y.buffer, f.scale.buffer, f.angle.buffer, f.desc.buffer]);
  } catch {
    self.postMessage({ id }); // normal reconstruction will retry this image
  }
};
