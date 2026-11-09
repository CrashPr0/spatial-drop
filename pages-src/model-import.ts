/** Convert OBJ or USDZ models to the GLB used by our viewer and transport. */
export async function importModel(file: File): Promise<File> {
  if (file.size > 15 * 1024 * 1024) throw new Error('That model is over 15 MB. Optimize it, then try again.');
  if (/\.glb$/i.test(file.name)) {
    const signature = new TextDecoder().decode(await file.slice(0, 4).arrayBuffer());
    if (signature !== 'glTF') throw new Error('This file is not a valid binary glTF model.');
    return file;
  }
  if (!/\.(obj|usdz)$/i.test(file.name)) throw new Error('Please choose a .glb, .obj or .usdz model.');
  const { GLTFExporter } = await import('three/addons/exporters/GLTFExporter.js');
  const isUsdz = /\.usdz$/i.test(file.name);
  const model = isUsdz
    ? new (await import('three/addons/loaders/USDLoader.js')).USDLoader().parse(await file.arrayBuffer())
    : new (await import('three/addons/loaders/OBJLoader.js')).OBJLoader().parse(await file.text());
  const textures = new Set<import('three').Texture>();
  try {
    let vertices = 0;
    model.traverse(node => {
      const mesh = node as import('three').Mesh;
      if (mesh.material) for (const material of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) {
        for (const value of Object.values(material)) if (value?.isTexture) textures.add(value);
      }
      const geometry = mesh.geometry;
      const positions = geometry?.getAttribute('position');
      if (!positions) return;
      vertices += positions.count;
      for (const value of positions.array) if (!Number.isFinite(value)) throw new Error('The model contains invalid vertex coordinates.');
    });
    if (!vertices) throw new Error('The model contains no geometry.');
    // USDLoader returns before its embedded images finish decoding.
    const deadline = performance.now() + 10000;
    while ([...textures].some(texture => !texture.image)) {
      if (performance.now() >= deadline) throw new Error('USDZ textures could not be decoded. Try exporting the model as GLB.');
      await new Promise(resolve => setTimeout(resolve, 25));
    }
    const result = await new GLTFExporter().parseAsync(model, { binary: true });
    if (!(result instanceof ArrayBuffer)) throw new Error('Model conversion failed.');
    if (result.byteLength > 15 * 1024 * 1024) throw new Error('The converted model exceeds 15 MB. Simplify the model and try again.');
    return new File([result], file.name.replace(/\.(obj|usdz)$/i, '.glb'), { type: 'model/gltf-binary' });
  } finally {
    for (const texture of textures) texture.dispose();
    model.traverse(node => {
      const mesh = node as import('three').Mesh;
      mesh.geometry?.dispose();
      if (mesh.material) for (const material of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) material.dispose();
    });
  }
}
