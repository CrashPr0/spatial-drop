import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { File } from "node:buffer";
import test from "node:test";
import {
  addBeamFrame,
  assembleBeamTransfer,
  createBeamFrame,
  createBeamTransfer,
  createCimbarFile,
  createReceiveBuffer,
  parseBeamFrame,
  verifyCimbarModel,
  type ReceiveBuffer,
} from "../pages-src/beam-codec";

async function demoTransfer() {
  const bytes = await readFile(new URL("../pages-src/demo/spartan-loop.glb", import.meta.url));
  const file = new File([bytes], "spartan-loop.glb", { type: "model/gltf-binary" });
  return { bytes, file, transfer: await createBeamTransfer(file) };
}

function receiveWithLoss(transfer: Awaited<ReturnType<typeof createBeamTransfer>>, start: number, dropEvery: number) {
  let buffer: ReceiveBuffer | null = null;
  const limit = start + transfer.fragments.length * 8;
  for (let sequence = start; sequence < limit; sequence += 1) {
    if (sequence % dropEvery === 1) continue;
    const frame = parseBeamFrame(createBeamFrame(transfer, sequence));
    assert.ok(frame);
    buffer ??= createReceiveBuffer(frame);
    addBeamFrame(buffer, frame);
    if (buffer.rank === buffer.total) return buffer;
  }
  assert.fail(`Recovery did not converge from sequence ${start} with 1/${dropEvery} frames dropped.`);
}

test("fountain frames recover a late-start transfer with periodic loss", async () => {
  const { bytes, transfer } = await demoTransfer();
  const starts = [0, 7, transfer.fragments.length + 10, transfer.fragments.length * 2 + 3];
  for (const start of starts) {
    for (const dropEvery of [3, 4, 5]) {
      const model = await assembleBeamTransfer(receiveWithLoss(transfer, start, dropEvery));
      assert.deepEqual(Buffer.from(model.bytes), bytes);
    }
  }
});

test("Cimbar envelope preserves and verifies the original GLB", async () => {
  const { bytes, transfer } = await demoTransfer();
  const wrapped = createCimbarFile(transfer);
  const model = await verifyCimbarModel(wrapped.name, await wrapped.arrayBuffer());
  assert.equal(model.name, "spartan-loop.glb");
  assert.deepEqual(Buffer.from(model.bytes), bytes);
  assert.equal(model.hash, transfer.hash);
});
