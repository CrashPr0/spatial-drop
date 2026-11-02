import assert from "node:assert/strict";
import test from "node:test";
import { bakeTextureAtlas, type ScanMesh, type Vec3 } from "../pages-src/scan-mesh";
import { projectPhotos, type PhotoView } from "../pages-src/scan-texture";

// Quads at depth z facing a camera at the origin that looks down +z (OpenCV axes).
function quads(list: { z: number; half: number; x?: number }[]): ScanMesh {
  const positions: number[] = [], normals: number[] = [], indices: number[] = [];
  for (const { z, half, x = 0 } of list) {
    const base = positions.length / 3;
    positions.push(x - half, -half, z, x + half, -half, z, x + half, half, z, x - half, half, z);
    for (let k = 0; k < 4; k++) normals.push(0, 0, -1);
    indices.push(base, base + 2, base + 1, base, base + 3, base + 2);
  }
  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    colors: new Float32Array(positions.length).fill(0.5),
    indices: new Uint32Array(indices),
    sampleColor: (_p: Vec3, out: Vec3) => { out[0] = out[1] = out[2] = 0; return out; },
    toWorld: (p: Vec3) => [p[0], p[1], p[2]],
    directionToWorld: (d: Vec3) => [d[0], d[1], d[2]],
    stats: { splatsUsed: 0, groundRemoved: false, voxels: [0, 0, 0], triangles: list.length * 2, sizeMeters: [1, 1, 0] },
  };
}

// a 200 x 200 photo whose red channel encodes the pixel column, green the row
const photo = (() => {
  const data = new Uint8ClampedArray(200 * 200 * 4);
  for (let y = 0; y < 200; y++) for (let x = 0; x < 200; x++) data.set([x, y, 255, 255], (y * 200 + x) * 4);
  return { data, width: 200, height: 200 };
})();
const view: PhotoView = {
  R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: [0, 0, 0], f: 200, featureWidth: 200, featureHeight: 200, k1: 0, k2: 0,
  load: async () => photo,
};

test("each texel takes the photo pixel its surface point projects to", async () => {
  const mesh = quads([{ z: 2, half: 0.5 }]);
  const atlas = bakeTextureAtlas(mesh, { size: 64 });
  const coverage = await projectPhotos(atlas, mesh, [view]);
  assert.equal(coverage, 1);
  let checked = 0;
  for (let i = 0; i < atlas.size * atlas.size; i++) {
    if (!atlas.texelUsed[i]) continue;
    const [X, Y, Z] = [atlas.texelPoints[i * 3], atlas.texelPoints[i * 3 + 1], atlas.texelPoints[i * 3 + 2]];
    const u = (200 * X) / Z + 100 - 0.5, v = (200 * Y) / Z + 100 - 0.5;
    if (u < 3 || v < 3 || u > 196 || v > 196) continue;
    assert.ok(Math.abs(atlas.pixels[i * 4] - u) <= 1, `red ${atlas.pixels[i * 4]} vs column ${u}`);
    assert.ok(Math.abs(atlas.pixels[i * 4 + 1] - v) <= 1, `green ${atlas.pixels[i * 4 + 1]} vs row ${v}`);
    checked++;
  }
  assert.ok(checked > 100);
});

test("surfaces facing away from every camera keep their splat color", async () => {
  const mesh = quads([{ z: 2, half: 0.5 }]);
  mesh.normals.fill(0).forEach((_, i, normals) => { if (i % 3 === 2) normals[i] = 1; });
  const atlas = bakeTextureAtlas(mesh, { size: 64 });
  assert.equal(await projectPhotos(atlas, mesh, [view]), 0);
});

test("occluded surface is not painted with the occluder's pixels", async () => {
  // a small quad in front hides the middle of the big one behind it
  const mesh = quads([{ z: 2, half: 0.5 }, { z: 1, half: 0.1 }]);
  const atlas = bakeTextureAtlas(mesh, { size: 128 });
  await projectPhotos(atlas, mesh, [view]);
  let hidden = 0, painted = 0;
  for (let i = 0; i < atlas.size * atlas.size; i++) {
    if (!atlas.texelUsed[i] || Math.abs(atlas.texelPoints[i * 3 + 2] - 2) > 1e-6) continue;
    const X = atlas.texelPoints[i * 3], Y = atlas.texelPoints[i * 3 + 1];
    // well inside the occluder's shadow (it covers |X|,|Y| < 0.2 at z = 2)
    if (Math.abs(X) > 0.15 || Math.abs(Y) > 0.15) continue;
    hidden++;
    if (atlas.pixels[i * 4 + 2] === 255) painted++;
  }
  assert.ok(hidden > 20, `hidden texels ${hidden}`);
  assert.equal(painted, 0);
});
