// A compact USDZ for Apple Quick Look. three's exporter writes every triangle
// corner as its own vertex in uncompressed text; USD can share points and
// still give each corner its own texture coordinate (faceVarying), which
// makes the file several times smaller, and it can carry a JPEG texture.

import { strToU8, zipSync } from "three/addons/libs/fflate.module.js";

export interface UsdzMesh {
  /** shared vertex positions and normals, meters, +Y up */
  positions: Float32Array;
  normals: Float32Array;
  /** triangle corners as indices into positions */
  indices: Uint32Array;
  /** one (u, v) per corner, glTF convention (v down from the image top) */
  uvs: Float32Array;
}

const number = (value: number, digits: number) => {
  const text = value.toFixed(digits).replace(/\.?0+$/, "");
  return text === "-0" ? "0" : text;
};

function tuples(values: ArrayLike<number>, width: number, digits: number, map?: (value: number, k: number) => number) {
  const parts: string[] = [];
  for (let i = 0; i < values.length; i += width) {
    const items: string[] = [];
    for (let k = 0; k < width; k++) items.push(number(map ? map(values[i + k], k) : values[i + k], digits));
    parts.push(`(${items.join(", ")})`);
  }
  return parts.join(", ");
}

export function buildUsda(mesh: UsdzMesh, texturePath: string) {
  const triangles = mesh.indices.length / 3;
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < mesh.positions.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      min[k] = Math.min(min[k], mesh.positions[i + k]);
      max[k] = Math.max(max[k], mesh.positions[i + k]);
    }
  }
  return `#usda 1.0
(
    customLayerData = {
        string creator = "Spatial Drop"
    }
    defaultPrim = "Root"
    metersPerUnit = 1
    upAxis = "Y"
)

def Xform "Root" (
    assetInfo = {
        string name = "Root"
    }
    kind = "component"
)
{
    token preliminary:anchoring:type = "plane"
    token preliminary:planeAnchoring:alignment = "horizontal"

    def Mesh "Scan" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        rel material:binding = </Root/Materials/Scan>
        float3[] extent = [(${min.map((v) => number(v, 5)).join(", ")}), (${max.map((v) => number(v, 5)).join(", ")})]
        int[] faceVertexCounts = [${new Array(triangles).fill(3).join(", ")}]
        int[] faceVertexIndices = [${Array.from(mesh.indices).join(", ")}]
        point3f[] points = [${tuples(mesh.positions, 3, 5)}]
        normal3f[] normals = [${tuples(mesh.normals, 3, 4)}] (
            interpolation = "vertex"
        )
        texCoord2f[] primvars:st = [${tuples(mesh.uvs, 2, 6, (value, k) => (k === 1 ? 1 - value : value))}] (
            interpolation = "faceVarying"
        )
        uniform token orientation = "rightHanded"
        uniform token subdivisionScheme = "none"
    }

    def Scope "Materials"
    {
        def Material "Scan"
        {
            token outputs:surface.connect = </Root/Materials/Scan/Surface.outputs:surface>

            def Shader "Surface"
            {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor.connect = </Root/Materials/Scan/Color.outputs:rgb>
                float inputs:metallic = 0
                float inputs:roughness = 0.85
                token outputs:surface
            }

            def Shader "Color"
            {
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @${texturePath}@
                float2 inputs:st.connect = </Root/Materials/Scan/Coordinates.outputs:result>
                token inputs:sourceColorSpace = "sRGB"
                token inputs:wrapS = "clamp"
                token inputs:wrapT = "clamp"
                float3 outputs:rgb
            }

            def Shader "Coordinates"
            {
                uniform token info:id = "UsdPrimvarReader_float2"
                float2 inputs:fallback = (0, 0)
                string inputs:varname = "st"
                float2 outputs:result
            }
        }
    }
}
`;
}

/** USDZ is an uncompressed zip whose file data starts on 64-byte boundaries. */
export function buildUsdz(mesh: UsdzMesh, jpeg: Uint8Array) {
  const texturePath = "textures/scan.jpg";
  const files: Record<string, Uint8Array> = {
    "model.usda": strToU8(buildUsda(mesh, texturePath)),
    [texturePath]: jpeg,
  };
  const entries: Record<string, Uint8Array | [Uint8Array, { extra: Record<number, Uint8Array> }]> = {};
  let offset = 0;
  for (const [name, data] of Object.entries(files)) {
    // local header (30 bytes) + name + the 4-byte header of our padding field
    offset += 30 + name.length + 4;
    const padding = (64 - (offset % 64)) % 64;
    entries[name] = [data, { extra: { 12345: new Uint8Array(padding) } }];
    offset += padding + data.length;
  }
  return zipSync(entries, { level: 0 });
}
