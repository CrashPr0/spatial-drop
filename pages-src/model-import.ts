/** Convert standalone OBJ geometry to the GLB used by our viewer and transport. */
export async function importModel(file: File): Promise<File> {
  if (file.size > 15 * 1024 * 1024) throw new Error('That model is over 15 MB. Optimize it, then try again.');
  if (/\.glb$/i.test(file.name)) {
    const signature = new TextDecoder().decode(await file.slice(0, 4).arrayBuffer());
    if (signature !== 'glTF') throw new Error('This file is not a valid binary glTF model.');
    return file;
  }
  if (!/\.obj$/i.test(file.name)) throw new Error('Please choose a .glb or .obj model.');
  const [{ OBJLoader }, { GLTFExporter }] = await Promise.all([
    import('three/addons/loaders/OBJLoader.js'),
    import('three/addons/exporters/GLTFExporter.js'),
  ]);
  const model = new OBJLoader().parse(await file.text());
  try {
    let vertices = 0;
    model.traverse(node => {
      const geometry = (node as import('three').Mesh).geometry;
      const positions = geometry?.getAttribute('position');
      if (!positions) return;
      vertices += positions.count;
      for (const value of positions.array) if (!Number.isFinite(value)) throw new Error('The OBJ contains invalid vertex coordinates.');
    });
    if (!vertices) throw new Error('The OBJ contains no geometry.');
    const result = await new GLTFExporter().parseAsync(model, { binary: true });
    if (!(result instanceof ArrayBuffer)) throw new Error('OBJ conversion failed.');
    if (result.byteLength > 15 * 1024 * 1024) throw new Error('The converted model exceeds 15 MB. Simplify the OBJ and try again.');
    return new File([result], file.name.replace(/\.obj$/i, '.glb'), { type: 'model/gltf-binary' });
  } finally {
    model.traverse(node => {
      const mesh = node as import('three').Mesh;
      mesh.geometry?.dispose();
      if (mesh.material) for (const material of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) material.dispose();
    });
  }
}
