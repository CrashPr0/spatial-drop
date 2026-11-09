import assert from 'node:assert/strict';
import test from 'node:test';
import { photoThumbnails } from '../pages-src/scan-thumbnails';

test('previews reuse decoded images and release failed, evicted and cleared URLs', (t) => {
  let created = 0;
  const revoked: string[] = [];
  t.mock.method(URL, 'createObjectURL', () => `blob:preview-${++created}`);
  t.mock.method(URL, 'revokeObjectURL', (url: string) => revoked.push(url));
  const children: Image[] = [];
  class Image extends EventTarget {
    src = '';
    alt = '';
    decoding = '';
    removeAttribute(name: string) { if (name === 'src') this.src = ''; }
    remove() { const i = children.indexOf(this); if (i >= 0) children.splice(i, 1); }
  }
  const container = {
    children,
    ownerDocument: { createElement: () => new Image() },
    insertBefore(image: Image, next: Image | null) {
      image.remove();
      children.splice(next ? children.indexOf(next) : children.length, 0, image);
    },
  };
  const update = photoThumbnails(container as unknown as HTMLElement);
  const files = Array.from({length: 13}, (_, i) => new File(['photo'], `${i}.jpg`));
  update(files.slice(0, 12));
  const first = children[0], second = children[1];
  first.dispatchEvent(new Event('load'));
  second.dispatchEvent(new Event('error'));
  assert.equal(revoked.length, 2);
  update(files.slice(0, 12));
  assert.equal(created, 12);
  assert.equal(children[0], first);
  update(files);
  assert.equal(created, 13);
  assert.equal(children.length, 12);
  assert.equal(children[0], second);
  assert.equal(first.src, '');
  update([]);
  assert.equal(children.length, 0);
  assert.equal(revoked.length, 13);
  assert.equal(new Set(revoked).size, 13);
  // A late load after removal must not revoke a URL twice.
  first.dispatchEvent(new Event('error'));
  assert.equal(revoked.length, 13);
});
