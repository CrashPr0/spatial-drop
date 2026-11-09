import assert from 'node:assert/strict';
import test from 'node:test';
import { PhotoPreparation } from '../pages-src/scan-preparation';
import { CaptureQueue } from '../pages-src/scan-queue';

class FakeWorker {
  onmessage: ((event: { data: { id: number; accepted: boolean } }) => void) | null = null;
  onerror: (() => void) | null = null;
  onmessageerror: (() => void) | null = null;
  messages: { id: number; file: number }[] = [];
  terminated = false;
  postMessage(message: { id: number; file: number }) { this.messages.push(message); }
  terminate() { this.terminated = true; }
  respond(id: number, accepted = true) { this.onmessage?.({ data: { id, accepted } }); }
}
const setup = () => {
  const workers: FakeWorker[] = [];
  const prep = new PhotoPreparation<{file: number}, {id: number; accepted: boolean}>(() => {
    const worker = new FakeWorker(); workers.push(worker); return worker as unknown as Worker;
  }, 100);
  return { workers, prep };
};
const tick = () => new Promise(resolve => setImmediate(resolve));

test('a timed-out photo does not bypass checks for queued photos', async (t) => {
  t.mock.timers.enable({apis: ['setTimeout']});
  const { workers, prep } = setup();
  const queue = new CaptureQueue<number>(async file => (await prep.request({file}))?.accepted ?? null, () => {});
  const first = queue.enqueue(1), second = queue.enqueue(2);
  t.mock.timers.tick(100);
  assert.equal(await first, null);
  await tick();
  assert.equal(workers[0].terminated, true);
  assert.equal(workers.length, 2);
  assert.equal(queue.pending, 1);
  // Late completion from the dead worker cannot finish its replacement.
  workers[0].respond(1);
  workers[1].respond(1);
  await tick();
  assert.equal(queue.pending, 1);
  workers[1].respond(2, false);
  assert.equal(await second, false);
  assert.equal(queue.pending, 0);
});

for (const event of ['onerror', 'onmessageerror'] as const) {
  test(`${event} preserves the photo and allows the next worker to succeed`, async () => {
    const {workers, prep} = setup();
    const first = prep.request({file: 1});
    workers[0][event]?.();
    assert.equal(await first, null);
    const second = prep.request({file: 2});
    workers[0][event]?.();
    assert.equal(workers[1].terminated, false);
    workers[1].respond(2);
    assert.equal((await second)?.accepted, true);
  });
}

test('successful checks reuse a worker; cancellation releases the active request', async () => {
  const {workers, prep} = setup();
  const first = prep.request({file: 1}); workers[0].respond(1); await first;
  const second = prep.request({file: 2});
  assert.equal(workers.length, 1);
  prep.cancel();
  assert.equal(await second, null);
  assert.equal(workers[0].terminated, true);
});

test('worker startup failure resolves unknown instead of stranding capture', async () => {
  const prep = new PhotoPreparation(() => { throw new Error('unavailable'); });
  assert.equal(await prep.request({}), null);
});
