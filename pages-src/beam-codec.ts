export const BEAM_MAX_BYTES = 128 * 1024;
export const BEAM_CHUNK_BYTES = 512;
export const BEAM_PROTOCOL = "SD2";

const BASE32_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
const MANIFEST_MAGIC = new TextEncoder().encode("SDB2");

export type BeamTransfer = {
  id: string;
  hash: string;
  name: string;
  size: number;
  transportSize: number;
  compressed: boolean;
  messageLength: number;
  fragments: Uint8Array[];
  originalBytes: Uint8Array;
};

export type ParsedBeamFrame = {
  id: string;
  sequence: number;
  total: number;
  fragmentSize: number;
  messageLength: number;
  bytes: Uint8Array;
};

type FountainEquation = {
  coefficients: Uint32Array;
  bytes: Uint8Array;
};

export type ReceiveBuffer = {
  id: string;
  total: number;
  fragmentSize: number;
  messageLength: number;
  rows: Map<number, FountainEquation>;
  seenSequences: Set<number>;
  rank: number;
  finalizing: boolean;
};

export type ReceivedModel = {
  bytes: Uint8Array;
  name: string;
  hash: string;
};

function bytesToHex(bytes: Uint8Array) {
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("").toUpperCase();
}

function hexToBytes(value: string) {
  if (value.length % 2 !== 0 || !/^[A-F0-9]+$/.test(value)) throw new Error("Invalid hexadecimal data.");
  const bytes = new Uint8Array(value.length / 2);
  for (let index = 0; index < bytes.length; index += 1) bytes[index] = Number.parseInt(value.slice(index * 2, index * 2 + 2), 16);
  return bytes;
}

export async function sha256Hex(bytes: ArrayBuffer | Uint8Array) {
  const source = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const copy = new Uint8Array(source.byteLength);
  copy.set(source);
  const digest = await crypto.subtle.digest("SHA-256", copy.buffer);
  return bytesToHex(new Uint8Array(digest));
}

function base32Encode(bytes: Uint8Array) {
  let output = "";
  let value = 0;
  let bits = 0;
  for (const byte of bytes) {
    value = (value << 8) | byte;
    bits += 8;
    while (bits >= 5) {
      output += BASE32_ALPHABET[(value >>> (bits - 5)) & 31];
      bits -= 5;
    }
    value &= (1 << bits) - 1;
  }
  if (bits > 0) output += BASE32_ALPHABET[(value << (5 - bits)) & 31];
  return output;
}

function base32Decode(value: string) {
  if (!/^[A-Z2-7]+$/.test(value)) throw new Error("Invalid Base32 data.");
  const output = new Uint8Array(Math.floor((value.length * 5) / 8));
  let accumulator = 0;
  let bits = 0;
  let offset = 0;
  for (const character of value) {
    accumulator = (accumulator << 5) | BASE32_ALPHABET.indexOf(character);
    bits += 5;
    if (bits >= 8) {
      output[offset] = (accumulator >>> (bits - 8)) & 255;
      offset += 1;
      bits -= 8;
    }
    accumulator &= (1 << bits) - 1;
  }
  return output;
}

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let index = 0; index < 256; index += 1) {
    let value = index;
    for (let bit = 0; bit < 8; bit += 1) value = (value & 1) ? (0xedb88320 ^ (value >>> 1)) : (value >>> 1);
    table[index] = value >>> 0;
  }
  return table;
})();

function crc32(bytes: Uint8Array) {
  let crc = 0xffffffff;
  for (const byte of bytes) crc = CRC_TABLE[(crc ^ byte) & 255] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}

async function streamTransform(bytes: Uint8Array, stream: CompressionStream | DecompressionStream) {
  const copy = new Uint8Array(bytes.length);
  copy.set(bytes);
  const output = new Response(stream.readable).arrayBuffer();
  const writer = stream.writable.getWriter();
  await writer.write(copy.buffer);
  await writer.close();
  return new Uint8Array(await output);
}

async function gzip(bytes: Uint8Array) {
  return streamTransform(bytes, new CompressionStream("gzip"));
}

async function gunzip(bytes: Uint8Array) {
  return streamTransform(bytes, new DecompressionStream("gzip"));
}

function buildManifest(name: string, size: number, hash: string, body: Uint8Array, compressed: boolean) {
  const nameBytes = new TextEncoder().encode(name.slice(0, 80));
  const headerLength = 42 + nameBytes.length;
  const message = new Uint8Array(headerLength + body.length);
  message.set(MANIFEST_MAGIC, 0);
  message[4] = compressed ? 1 : 0;
  new DataView(message.buffer).setUint32(5, size, false);
  message.set(hexToBytes(hash), 9);
  message[41] = nameBytes.length;
  message.set(nameBytes, 42);
  message.set(body, headerLength);
  return message;
}

function hasGlbSignature(bytes: Uint8Array) {
  return bytes.length >= 4 && new TextDecoder().decode(bytes.slice(0, 4)) === "glTF";
}

export async function createBeamTransfer(file: File): Promise<BeamTransfer> {
  if (!file.name.toLowerCase().endsWith(".glb")) throw new Error("QR Beam currently accepts binary .glb models.");
  if (file.size > BEAM_MAX_BYTES) throw new Error("This model is over the 128 KB optical transfer limit. Optimize it, then try again.");
  const originalBytes = new Uint8Array(await file.arrayBuffer());
  if (!hasGlbSignature(originalBytes)) throw new Error("This file is not a valid binary glTF model.");

  const hash = await sha256Hex(originalBytes);
  let body = originalBytes;
  let compressed = false;
  try {
    const candidate = await gzip(originalBytes);
    if (candidate.length + 24 < originalBytes.length) {
      body = candidate;
      compressed = true;
    }
  } catch {
    // CompressionStream is an optimization. Raw transfer remains interoperable.
  }

  const message = buildManifest(file.name, originalBytes.length, hash, body, compressed);
  const total = Math.ceil(message.length / BEAM_CHUNK_BYTES);
  if (total > 512) throw new Error("The compressed optical payload needs too many fragments.");
  const fragments = Array.from({ length: total }, (_, index) => {
    const fragment = new Uint8Array(BEAM_CHUNK_BYTES);
    fragment.set(message.slice(index * BEAM_CHUNK_BYTES, (index + 1) * BEAM_CHUNK_BYTES));
    return fragment;
  });

  return {
    id: hash.slice(0, 12),
    hash,
    name: file.name,
    size: originalBytes.length,
    transportSize: body.length,
    compressed,
    messageLength: message.length,
    fragments,
    originalBytes,
  };
}

function seededRandom(id: string, sequence: number) {
  // Xorshift generators are linear over GF(2), which is exactly the field used
  // by the decoder. That can make recovery rows correlated. This integer mixer
  // deliberately uses addition and multiplication to break that relationship.
  let state = (Number.parseInt(id.slice(0, 8), 16) + Math.imul(sequence + 1, 0x9e3779b9)) >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return (value ^ (value >>> 14)) >>> 0;
  };
}

export function fountainIndexes(id: string, sequence: number, total: number) {
  if (sequence < total) return [sequence];
  const random = seededRandom(id, sequence);
  const indexes: number[] = [];
  for (let index = 0; index < total; index += 1) {
    if ((random() & 1) === 1) indexes.push(index);
  }
  if (indexes.length === 0) indexes.push(random() % total);
  return indexes;
}

export function createBeamFrame(transfer: BeamTransfer, sequence: number) {
  const payload = new Uint8Array(BEAM_CHUNK_BYTES);
  for (const index of fountainIndexes(transfer.id, sequence, transfer.fragments.length)) {
    const fragment = transfer.fragments[index];
    for (let offset = 0; offset < payload.length; offset += 1) payload[offset] ^= fragment[offset];
  }
  return [
    BEAM_PROTOCOL,
    transfer.id,
    sequence.toString(36).toUpperCase(),
    transfer.fragments.length.toString(36).toUpperCase(),
    BEAM_CHUNK_BYTES.toString(36).toUpperCase(),
    transfer.messageLength.toString(36).toUpperCase(),
    crc32(payload).toString(16).padStart(8, "0").toUpperCase(),
    base32Encode(payload),
  ].join(":");
}

export function parseBeamFrame(value: string): ParsedBeamFrame | null {
  const parts = value.split(":");
  if (parts.length !== 8 || parts[0] !== BEAM_PROTOCOL) return null;
  const [, id, sequenceValue, totalValue, fragmentSizeValue, messageLengthValue, checksum, payload] = parts;
  if (!/^[A-F0-9]{12}$/.test(id) || !/^[A-F0-9]{8}$/.test(checksum)) return null;
  if (![sequenceValue, totalValue, fragmentSizeValue, messageLengthValue].every((part) => /^[0-9A-Z]+$/.test(part))) return null;
  const sequence = Number.parseInt(sequenceValue, 36);
  const total = Number.parseInt(totalValue, 36);
  const fragmentSize = Number.parseInt(fragmentSizeValue, 36);
  const messageLength = Number.parseInt(messageLengthValue, 36);
  if (![sequence, total, fragmentSize, messageLength].every(Number.isSafeInteger)) return null;
  if (sequence < 0 || sequence > 0xffffffff || total < 1 || total > 512) return null;
  if (fragmentSize !== BEAM_CHUNK_BYTES || messageLength < 42 || messageLength > total * fragmentSize) return null;
  try {
    const bytes = base32Decode(payload);
    if (bytes.length !== fragmentSize || crc32(bytes) !== Number.parseInt(checksum, 16)) return null;
    return { id, sequence, total, fragmentSize, messageLength, bytes };
  } catch {
    return null;
  }
}

export function createReceiveBuffer(frame: ParsedBeamFrame): ReceiveBuffer {
  return {
    id: frame.id,
    total: frame.total,
    fragmentSize: frame.fragmentSize,
    messageLength: frame.messageLength,
    rows: new Map(),
    seenSequences: new Set(),
    rank: 0,
    finalizing: false,
  };
}

function coefficientBit(coefficients: Uint32Array, index: number) {
  return (coefficients[index >>> 5] & (1 << (index & 31))) !== 0;
}

function xorEquation(target: FountainEquation, source: FountainEquation) {
  for (let index = 0; index < target.coefficients.length; index += 1) target.coefficients[index] ^= source.coefficients[index];
  for (let index = 0; index < target.bytes.length; index += 1) target.bytes[index] ^= source.bytes[index];
}

export function addBeamFrame(buffer: ReceiveBuffer, frame: ParsedBeamFrame) {
  if (frame.id !== buffer.id || frame.total !== buffer.total || frame.fragmentSize !== buffer.fragmentSize || frame.messageLength !== buffer.messageLength) return false;
  if (buffer.finalizing || buffer.seenSequences.has(frame.sequence)) return false;
  buffer.seenSequences.add(frame.sequence);

  const equation: FountainEquation = {
    coefficients: new Uint32Array(Math.ceil(buffer.total / 32)),
    bytes: frame.bytes.slice(),
  };
  for (const index of fountainIndexes(frame.id, frame.sequence, frame.total)) equation.coefficients[index >>> 5] |= 1 << (index & 31);

  for (let pivot = 0; pivot < buffer.total; pivot += 1) {
    if (!coefficientBit(equation.coefficients, pivot)) continue;
    const existing = buffer.rows.get(pivot);
    if (existing) {
      xorEquation(equation, existing);
    } else {
      buffer.rows.set(pivot, equation);
      buffer.rank += 1;
      return true;
    }
  }
  return false;
}

function solveFountain(buffer: ReceiveBuffer) {
  if (buffer.rank !== buffer.total) throw new Error("More recovery frames are required.");
  const solved = new Array<Uint8Array>(buffer.total);
  for (let pivot = buffer.total - 1; pivot >= 0; pivot -= 1) {
    const row = buffer.rows.get(pivot);
    if (!row) throw new Error("The fountain matrix is incomplete.");
    const bytes = row.bytes.slice();
    for (let index = pivot + 1; index < buffer.total; index += 1) {
      if (!coefficientBit(row.coefficients, index)) continue;
      const known = solved[index];
      if (!known) throw new Error("The fountain matrix could not be resolved.");
      for (let offset = 0; offset < bytes.length; offset += 1) bytes[offset] ^= known[offset];
    }
    solved[pivot] = bytes;
  }
  const message = new Uint8Array(buffer.messageLength);
  let offset = 0;
  for (const fragment of solved) {
    const length = Math.min(fragment.length, message.length - offset);
    message.set(fragment.slice(0, length), offset);
    offset += length;
  }
  return message;
}

async function unpackManifest(message: Uint8Array): Promise<ReceivedModel> {
  if (message.length < 42 || !MANIFEST_MAGIC.every((byte, index) => message[index] === byte)) throw new Error("The reconstructed manifest is invalid.");
  const compressed = message[4] === 1;
  if (message[4] > 1) throw new Error("The reconstructed compression mode is not supported.");
  const size = new DataView(message.buffer, message.byteOffset, message.byteLength).getUint32(5, false);
  if (size < 1 || size > BEAM_MAX_BYTES) throw new Error("The reconstructed model size is invalid.");
  const hash = bytesToHex(message.slice(9, 41));
  const nameLength = message[41];
  const bodyOffset = 42 + nameLength;
  if (bodyOffset > message.length) throw new Error("The reconstructed filename is invalid.");
  const name = new TextDecoder().decode(message.slice(42, bodyOffset)).replace(/[\\/]/g, "_").slice(0, 80) || "spatial-drop.glb";
  const body = message.slice(bodyOffset);
  const bytes = compressed ? await gunzip(body) : body;
  if (bytes.length !== size) throw new Error("The reconstructed model size does not match its manifest.");
  if (!hasGlbSignature(bytes)) throw new Error("The reconstructed data is not a valid GLB.");
  if (await sha256Hex(bytes) !== hash) throw new Error("SHA-256 verification failed. Keep scanning recovery frames.");
  return { bytes, name, hash };
}

export async function assembleBeamTransfer(buffer: ReceiveBuffer) {
  return unpackManifest(solveFountain(buffer));
}

export function createCimbarFile(transfer: BeamTransfer) {
  const safeName = transfer.name.replace(/[^a-zA-Z0-9._-]+/g, "_").slice(-64) || "spatial-drop.glb";
  const encodedName = `SD-${transfer.hash}-${transfer.size.toString(36).toUpperCase()}-${safeName}`;
  const copy = new Uint8Array(transfer.originalBytes.length);
  copy.set(transfer.originalBytes);
  return new File([copy.buffer], encodedName, { type: "model/gltf-binary" });
}

export async function verifyCimbarModel(name: string, buffer: ArrayBuffer): Promise<ReceivedModel> {
  const match = /^SD-([A-F0-9]{64})-([A-Z0-9]+)-(.+\.glb)$/i.exec(name);
  if (!match) throw new Error("This Cimbar payload is not a Spatial Drop GLB.");
  const [, expectedHashValue, sizeValue, originalName] = match;
  const expectedHash = expectedHashValue.toUpperCase();
  const expectedSize = Number.parseInt(sizeValue, 36);
  const bytes = new Uint8Array(buffer);
  if (bytes.length !== expectedSize || bytes.length > BEAM_MAX_BYTES) throw new Error("The Cimbar model size does not match its envelope.");
  if (!hasGlbSignature(bytes)) throw new Error("The Cimbar payload is not a valid GLB.");
  if (await sha256Hex(bytes) !== expectedHash) throw new Error("The Cimbar model failed SHA-256 verification.");
  return { bytes, name: originalName.replace(/[\\/]/g, "_").slice(0, 80), hash: expectedHash };
}
