import assert from 'node:assert/strict';
import test from 'node:test';
import { LatestScanSave } from '../pages-src/scan-save';
const tick = () => new Promise(resolve => setImmediate(resolve));

test('slow storage writes only the active and newest waiting scan', async () => {
  const writes: number[] = [], finish: (() => void)[] = [];
  const saver = new LatestScanSave<number>(value => {
    writes.push(value); return new Promise<void>(resolve => finish.push(resolve));
  }, () => assert.fail('unexpected failure'));
  saver.save(1); saver.save(2); saver.save(3);
  assert.deepEqual(writes, [1]);
  finish[0](); await tick();
  assert.deepEqual(writes, [1, 3]);
  finish[1](); await tick();
  saver.save(4);
  assert.deepEqual(writes, [1, 3, 4]);
  finish[2]();
});

test('clearing a scan supersedes pending photo backups', async () => {
  const writes: number[][] = [], finish: (() => void)[] = [];
  const saver = new LatestScanSave<number[]>(value => {
    writes.push(value); return new Promise<void>(resolve => finish.push(resolve));
  }, () => assert.fail('unexpected failure'));
  saver.save([1]); saver.save([1, 2]); saver.save([]);
  finish[0](); await tick();
  assert.deepEqual(writes, [[1], []]);
  finish[1]();
});

test('failed storage reports the error and still writes the latest snapshot', async () => {
  const writes: number[] = [];
  let failures = 0;
  const saver = new LatestScanSave<number>(async value => {
    writes.push(value); if (value === 1) throw new Error('quota');
  }, () => { failures++; });
  saver.save(1); saver.save(2);
  await tick();
  assert.equal(failures, 1);
  assert.deepEqual(writes, [1, 2]);
});
