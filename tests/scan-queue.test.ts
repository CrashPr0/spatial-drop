import assert from 'node:assert/strict';
import test from 'node:test';
import { CaptureQueue } from '../pages-src/scan-queue';
const tick = () => new Promise(resolve => setImmediate(resolve));
test('capture queue bounds memory and serializes worker access', async () => {
  const started: number[] = [], finish: ((value: boolean) => void)[] = [];
  const q = new CaptureQueue<number>(n => { started.push(n); return new Promise(resolve => finish.push(resolve)); }, () => {});
  const a=q.enqueue(1), b=q.enqueue(2), c=q.enqueue(3);
  assert.equal(q.enqueue(4), null);
  assert.deepEqual(started,[1]);
  finish[0](true); assert.equal(await a,true); await tick();
  assert.deepEqual(started,[1,2]);
  finish[1](false); assert.equal(await b,false); await tick();
  finish[2](true); assert.equal(await c,true); assert.equal(q.pending,0);
});
test('reset cancels old candidates without consuming a new scan queue', async () => {
  const finish: ((value: boolean) => void)[]=[];
  const q=new CaptureQueue<number>(()=>new Promise(resolve=>finish.push(resolve)),()=>{});
  const old=q.enqueue(1), queued=q.enqueue(2); q.reset();
  assert.equal(await old,null); assert.equal(await queued,null);
  const next=q.enqueue(3); finish[0](true); await tick();
  assert.equal(q.pending,1); finish[1](true); assert.equal(await next,true);
});
test('worker errors preserve an unknown candidate and release the queue', async () => {
  const q=new CaptureQueue<number>(async()=>{throw new Error('worker failed');},()=>{});
  assert.equal(await q.enqueue(1),null);assert.equal(q.pending,0);
});
