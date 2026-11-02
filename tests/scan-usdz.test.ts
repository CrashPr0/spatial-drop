import assert from "node:assert/strict";
import test from "node:test";
import { unzipSync } from "three/addons/libs/fflate.module.js";
import { buildUsdz } from "../pages-src/scan-usdz";

// a unit square of two triangles sharing an edge
const square = {
  positions: new Float32Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]),
  normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]),
  indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
  uvs: new Float32Array([0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0]),
};
const jpeg = new Uint8Array([0xff, 0xd8, 0xff, 0xe0, 1, 2, 3, 0xff, 0xd9]);

test("USDZ entries are stored uncompressed on 64-byte boundaries, model first", () => {
  const zip = buildUsdz(square, jpeg);
  const view = new DataView(zip.buffer, zip.byteOffset, zip.byteLength);
  const names: string[] = [];
  for (let offset = 0; view.getUint32(offset, true) === 0x04034b50;) {
    const method = view.getUint16(offset + 8, true);
    const size = view.getUint32(offset + 18, true);
    const nameLength = view.getUint16(offset + 26, true);
    const extraLength = view.getUint16(offset + 28, true);
    names.push(new TextDecoder().decode(zip.subarray(offset + 30, offset + 30 + nameLength)));
    const dataStart = offset + 30 + nameLength + extraLength;
    assert.equal(method, 0, `${names.at(-1)} is compressed`);
    assert.equal(dataStart % 64, 0, `${names.at(-1)} data starts at ${dataStart}`);
    offset = dataStart + size;
  }
  assert.deepEqual(names, ["model.usda", "textures/scan.jpg"]);
  assert.deepEqual(unzipSync(zip)["textures/scan.jpg"], jpeg);
});

test("USDA shares points and gives every corner its own flipped texture coordinate", () => {
  const usda = new TextDecoder().decode(unzipSync(buildUsdz(square, jpeg))["model.usda"]);
  assert.match(usda, /int\[\] faceVertexIndices = \[0, 1, 2, 0, 2, 3\]/);
  assert.match(usda, /point3f\[\] points = \[\(0, 0, 0\), \(1, 0, 0\), \(1, 1, 0\), \(0, 1, 0\)\]/);
  // glTF v (down from the top) becomes USD t (up from the bottom)
  assert.match(usda, /texCoord2f\[\] primvars:st = \[\(0, 0\), \(1, 0\), \(1, 1\), \(0, 0\), \(1, 1\), \(0, 1\)\] \(\s*interpolation = "faceVarying"/);
  assert.match(usda, /asset inputs:file = @textures\/scan\.jpg@/);
  assert.match(usda, /float3\[\] extent = \[\(0, 0, 0\), \(1, 1, 0\)\]/);
});
