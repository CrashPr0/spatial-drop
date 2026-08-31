import QRCode from "qrcode";
import jsQR from "jsqr";

export const BEAM_MAX_BYTES = 128 * 1024;
export const BEAM_CHUNK_BYTES = 640;
const FRAME_DELAY_MS = 220;
const COLOR_FRAME_DELAY_MS = 360;
const COLOR_LAYERS = 3;
const QR_RENDER_WIDTH = 720;
const PROTOCOL = "SD1";

export type BeamTransport = "mono" | "color3";

export type BeamTransfer = {
  id: string;
  hash: string;
  name: string;
  size: number;
  frames: string[];
};

export type ParsedBeamFrame = {
  id: string;
  hash: string;
  name: string;
  size: number;
  index: number;
  total: number;
  bytes: Uint8Array;
};

type ReceiveBuffer = Omit<ParsedBeamFrame, "index" | "bytes"> & {
  chunks: Map<number, Uint8Array>;
  finalizing: boolean;
};

let scannerStream: MediaStream | null = null;
let scannerAnimation = 0;
let receivedObjectUrl = "";

function formatBytes(bytes: number) {
  return bytes < 1024 ? `${bytes} B` : `${(bytes / 1024).toFixed(1)} KB`;
}

function bytesToBase64(bytes: Uint8Array) {
  let binary = "";
  for (let index = 0; index < bytes.length; index += 1) binary += String.fromCharCode(bytes[index]);
  return btoa(binary);
}

function base64ToBytes(value: string) {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return bytes;
}

async function sha256Hex(bytes: ArrayBuffer | Uint8Array) {
  const source = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const copy = new Uint8Array(source.byteLength);
  copy.set(source);
  const digest = await crypto.subtle.digest("SHA-256", copy.buffer);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

export async function createBeamTransfer(file: File): Promise<BeamTransfer> {
  if (!file.name.toLowerCase().endsWith(".glb")) throw new Error("QR Beam currently accepts binary .glb models.");
  if (file.size > BEAM_MAX_BYTES) throw new Error("This model is over the 128 KB optical transfer limit. Optimize it, then try again.");
  const buffer = await file.arrayBuffer();
  if (new TextDecoder().decode(buffer.slice(0, 4)) !== "glTF") throw new Error("This file is not a valid binary glTF model.");

  const bytes = new Uint8Array(buffer);
  const hash = await sha256Hex(bytes);
  const id = hash.slice(0, 12);
  const name = encodeURIComponent(file.name.slice(0, 80));
  const total = Math.ceil(bytes.length / BEAM_CHUNK_BYTES);
  const frames: string[] = [];

  for (let index = 0; index < total; index += 1) {
    const chunk = bytes.slice(index * BEAM_CHUNK_BYTES, (index + 1) * BEAM_CHUNK_BYTES);
    frames.push([PROTOCOL, id, index, total, bytes.length, hash, name, bytesToBase64(chunk)].join("|"));
  }
  return { id, hash, name: file.name, size: bytes.length, frames };
}

export function parseBeamFrame(value: string): ParsedBeamFrame | null {
  const parts = value.split("|");
  if (parts.length !== 8 || parts[0] !== PROTOCOL) return null;
  const [, id, indexValue, totalValue, sizeValue, hash, encodedName, payload] = parts;
  const index = Number(indexValue);
  const total = Number(totalValue);
  const size = Number(sizeValue);
  if (!/^[a-f0-9]{12}$/.test(id) || !/^[a-f0-9]{64}$/.test(hash)) return null;
  if (!Number.isInteger(index) || !Number.isInteger(total) || !Number.isInteger(size)) return null;
  if (index < 0 || total < 1 || index >= total || total > 512 || size < 1 || size > BEAM_MAX_BYTES) return null;
  try {
    const bytes = base64ToBytes(payload);
    if (bytes.length < 1 || bytes.length > BEAM_CHUNK_BYTES) return null;
    return { id, hash, name: decodeURIComponent(encodedName), size, index, total, bytes };
  } catch {
    return null;
  }
}

export async function assembleBeamTransfer(buffer: ReceiveBuffer) {
  if (buffer.chunks.size !== buffer.total) throw new Error("Some transfer frames are still missing.");
  const bytes = new Uint8Array(buffer.size);
  let offset = 0;
  for (let index = 0; index < buffer.total; index += 1) {
    const chunk = buffer.chunks.get(index);
    if (!chunk) throw new Error(`Frame ${index + 1} is missing.`);
    const remaining = bytes.length - offset;
    bytes.set(chunk.slice(0, remaining), offset);
    offset += Math.min(chunk.length, remaining);
  }
  if (offset !== buffer.size) throw new Error("The reconstructed model size does not match its manifest.");
  if (await sha256Hex(bytes) !== buffer.hash) throw new Error("Integrity check failed. Scan another loop and try again.");
  if (new TextDecoder().decode(bytes.slice(0, 4)) !== "glTF") throw new Error("The reconstructed data is not a valid GLB.");
  return bytes;
}

function setMessage(element: HTMLElement, message = "") {
  element.textContent = message;
  element.hidden = !message;
}

function screenFrameCount(transfer: BeamTransfer, transport: BeamTransport) {
  return transport === "color3" ? Math.ceil(transfer.frames.length / COLOR_LAYERS) : transfer.frames.length;
}

function frameDelay(transport: BeamTransport) {
  return transport === "color3" ? COLOR_FRAME_DELAY_MS : FRAME_DELAY_MS;
}

async function renderColorFrame(canvas: HTMLCanvasElement, values: string[]) {
  const populated = Array.from({ length: COLOR_LAYERS }, (_, index) => values[index] ?? values[0]);
  const version = Math.max(...populated.map((value) => QRCode.create(value, { errorCorrectionLevel: "L" }).version));
  const layers = await Promise.all(populated.map(async (value) => {
    const layerCanvas = document.createElement("canvas");
    await QRCode.toCanvas(layerCanvas, value, {
      width: QR_RENDER_WIDTH,
      margin: 3,
      version,
      errorCorrectionLevel: "L",
      color: { dark: "#000000", light: "#ffffff" },
    });
    return layerCanvas.getContext("2d", { willReadFrequently: true })!.getImageData(0, 0, layerCanvas.width, layerCanvas.height);
  }));

  canvas.width = layers[0].width;
  canvas.height = layers[0].height;
  const context = canvas.getContext("2d")!;
  const output = context.createImageData(canvas.width, canvas.height);
  for (let offset = 0; offset < output.data.length; offset += 4) {
    output.data[offset] = layers[0].data[offset];
    output.data[offset + 1] = layers[1].data[offset + 1];
    output.data[offset + 2] = layers[2].data[offset + 2];
    output.data[offset + 3] = 255;
  }
  context.putImageData(output, 0, 0);
}

function isolateColorChannel(image: ImageData, channel: 0 | 1 | 2) {
  const isolated = new Uint8ClampedArray(image.data.length);
  for (let offset = 0; offset < image.data.length; offset += 4) {
    const value = image.data[offset + channel];
    isolated[offset] = value;
    isolated[offset + 1] = value;
    isolated[offset + 2] = value;
    isolated[offset + 3] = 255;
  }
  return isolated;
}

export function stopBeamScanner() {
  cancelAnimationFrame(scannerAnimation);
  scannerAnimation = 0;
  scannerStream?.getTracks().forEach((track) => track.stop());
  scannerStream = null;
  const video = document.querySelector<HTMLVideoElement>("#beam-video");
  const scanCanvas = document.querySelector<HTMLCanvasElement>("#beam-scan-canvas");
  const viewport = document.querySelector<HTMLElement>(".scanner-viewport");
  if (video) {
    video.pause();
    video.srcObject = null;
    video.removeAttribute("src");
    video.load();
  }
  if (scanCanvas) {
    scanCanvas.width = 1;
    scanCanvas.height = 1;
    scanCanvas.getContext("2d")?.clearRect(0, 0, 1, 1);
  }
  viewport?.classList.remove("camera-active");
  const startButton = document.querySelector<HTMLButtonElement>("#scanner-start-button");
  const stopButton = document.querySelector<HTMLButtonElement>("#scanner-stop-button");
  const idle = document.querySelector<HTMLElement>("#scanner-idle");
  if (startButton) startButton.hidden = false;
  if (stopButton) stopButton.hidden = true;
  if (idle) idle.hidden = false;
}

export function setupBeamLab() {
  const beamDropZone = document.querySelector<HTMLElement>("#beam-drop-zone")!;
  const beamFileInput = document.querySelector<HTMLInputElement>("#beam-file-input")!;
  const beamFileTitle = document.querySelector<HTMLElement>("#beam-file-title")!;
  const beamFileDetail = document.querySelector<HTMLElement>("#beam-file-detail")!;
  const beamError = document.querySelector<HTMLElement>("#beam-error")!;
  const beamDemoButton = document.querySelector<HTMLButtonElement>("#beam-demo-button")!;
  const beamStartButton = document.querySelector<HTMLButtonElement>("#beam-start-button")!;
  const beamPauseButton = document.querySelector<HTMLButtonElement>("#beam-pause-button")!;
  const beamFullscreenButton = document.querySelector<HTMLButtonElement>("#beam-fullscreen-button")!;
  const beamDisplayCard = document.querySelector<HTMLElement>("#beam-display-card")!;
  const beamScreen = document.querySelector<HTMLElement>("#beam-screen")!;
  const beamCanvas = document.querySelector<HTMLCanvasElement>("#beam-canvas")!;
  const beamFrameLabel = document.querySelector<HTMLElement>("#beam-frame-label")!;
  const beamLoopLabel = document.querySelector<HTMLElement>("#beam-loop-label")!;
  const beamTransferSummary = document.querySelector<HTMLElement>("#beam-transfer-summary")!;
  const receiverLinkCanvas = document.querySelector<HTMLCanvasElement>("#receiver-link-qr")!;
  const transportInputs = Array.from(document.querySelectorAll<HTMLInputElement>('input[name="beam-transport"]'));
  const transportNote = document.querySelector<HTMLElement>("#beam-transport-note")!;
  const transportSpec = document.querySelector<HTMLElement>("#beam-transport-spec")!;
  const chunkSpec = document.querySelector<HTMLElement>("#beam-chunk-spec")!;
  let transfer: BeamTransfer | null = null;
  let beamTimer = 0;
  let frameIndex = 0;
  let loop = 1;
  let playing = false;
  let transport: BeamTransport = new URL(window.location.href).searchParams.get("beam") === "color3" ? "color3" : "mono";

  async function renderReceiverLink() {
    const receiverUrl = new URL(window.location.href);
    receiverUrl.search = "";
    if (transport === "color3") receiverUrl.searchParams.set("beam", "color3");
    receiverUrl.hash = "receive";
    await QRCode.toCanvas(receiverLinkCanvas, receiverUrl.toString(), {
      width: 420,
      margin: 3,
      errorCorrectionLevel: "H",
      color: { dark: "#0055A2", light: "#ffffff" },
    });
  }

  function updateTransferSummary() {
    if (!transfer) return;
    const visibleFrames = screenFrameCount(transfer, transport);
    const seconds = Math.ceil((visibleFrames * frameDelay(transport)) / 1000);
    beamFileDetail.textContent = transport === "color3"
      ? `${formatBytes(transfer.size)} · ${transfer.frames.length} data frames in ${visibleFrames} color frames`
      : `${formatBytes(transfer.size)} · ${visibleFrames} optical frames`;
    beamTransferSummary.textContent = transport === "color3"
      ? `${transfer.name} · ${formatBytes(transfer.size)} · ${visibleFrames} RGB-multiplexed frames · approximately ${seconds} seconds per loop. Each color channel carries a different QR.`
      : `${transfer.name} · ${formatBytes(transfer.size)} · approximately ${seconds} seconds per loop. Missed frames are recovered automatically on the next pass.`;
  }

  function updateTransportUi() {
    transportInputs.forEach((input) => { input.checked = input.value === transport; });
    const isColor = transport === "color3";
    transportNote.textContent = isColor
      ? "Experimental: scan the newly generated receiver code. Bright displays and a steady camera work best."
      : "Works with ordinary displays and the Spatial Drop receiver.";
    transportSpec.textContent = isColor ? "RGB multiplex" : "Animated QR";
    chunkSpec.textContent = isColor ? "3 × 640 bytes" : "640 bytes";
    beamScreen.classList.toggle("color3", isColor);
    beamCanvas.setAttribute("aria-label", isColor ? "Animated RGB multiplexed QR model data stream" : "Animated QR model data stream");
    updateTransferSummary();
    void renderReceiverLink();
  }

  async function renderCurrentFrame(scheduleNext = false) {
    if (!transfer) return;
    try {
      const visibleFrames = screenFrameCount(transfer, transport);
      if (transport === "color3") {
        const start = frameIndex * COLOR_LAYERS;
        await renderColorFrame(beamCanvas, transfer.frames.slice(start, start + COLOR_LAYERS));
      } else {
        await QRCode.toCanvas(beamCanvas, transfer.frames[frameIndex], {
          width: QR_RENDER_WIDTH,
          margin: 3,
          errorCorrectionLevel: "L",
          color: { dark: "#001f3f", light: "#ffffff" },
        });
      }
      beamFrameLabel.textContent = transport === "color3"
        ? `COLOR FRAME ${frameIndex + 1} / ${visibleFrames} · 3 DATA LAYERS`
        : `FRAME ${frameIndex + 1} / ${visibleFrames}`;
      beamLoopLabel.textContent = `LOOP ${loop}`;
      if (!scheduleNext || !playing) return;
      frameIndex += 1;
      if (frameIndex >= visibleFrames) {
        frameIndex = 0;
        loop += 1;
      }
      beamTimer = window.setTimeout(() => void renderCurrentFrame(true), frameDelay(transport));
    } catch (reason) {
      playing = false;
      beamPauseButton.textContent = "Resume";
      setMessage(beamError, reason instanceof Error ? reason.message : "Could not render the optical stream.");
    }
  }

  function stopBeam() {
    playing = false;
    window.clearTimeout(beamTimer);
    beamPauseButton.textContent = "Resume";
  }

  async function acceptBeamFile(file?: File) {
    if (!file) return;
    stopBeam();
    setMessage(beamError);
    beamStartButton.disabled = true;
    beamFileTitle.textContent = file.name;
    beamFileDetail.textContent = "Preparing manifest and integrity hash…";
    try {
      transfer = await createBeamTransfer(file);
      frameIndex = 0;
      loop = 1;
      updateTransferSummary();
      beamStartButton.disabled = false;
      beamStartButton.querySelector("span")!.textContent = "Start optical beam";
      await renderCurrentFrame();
    } catch (reason) {
      transfer = null;
      beamFileDetail.textContent = "Could not prepare this model";
      setMessage(beamError, reason instanceof Error ? reason.message : "Could not prepare this model.");
    }
  }

  beamDropZone.addEventListener("click", () => beamFileInput.click());
  beamDropZone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") beamFileInput.click();
  });
  beamDropZone.addEventListener("dragover", (event) => { event.preventDefault(); beamDropZone.classList.add("dragging"); });
  beamDropZone.addEventListener("dragleave", () => beamDropZone.classList.remove("dragging"));
  beamDropZone.addEventListener("drop", (event) => {
    event.preventDefault();
    beamDropZone.classList.remove("dragging");
    void acceptBeamFile(event.dataTransfer?.files[0]);
  });
  beamFileInput.addEventListener("change", () => void acceptBeamFile(beamFileInput.files?.[0]));
  transportInputs.forEach((input) => input.addEventListener("change", () => {
    stopBeam();
    transport = input.value === "color3" ? "color3" : "mono";
    frameIndex = 0;
    loop = 1;
    updateTransportUi();
    if (transfer) void renderCurrentFrame();
  }));
  beamDemoButton.addEventListener("click", async () => {
    setMessage(beamError);
    beamDemoButton.disabled = true;
    try {
      const response = await fetch(new URL("./demo/spartan-loop.glb", import.meta.url));
      if (!response.ok) throw new Error("The demo model could not be loaded.");
      const blob = await response.blob();
      await acceptBeamFile(new File([blob], "spartan-loop.glb", { type: "model/gltf-binary" }));
    } catch (reason) {
      setMessage(beamError, reason instanceof Error ? reason.message : "The demo model could not be loaded.");
    } finally {
      beamDemoButton.disabled = false;
    }
  });
  beamStartButton.addEventListener("click", () => {
    if (!transfer) return;
    stopBeam();
    playing = true;
    beamPauseButton.textContent = "Pause";
    beamDisplayCard.hidden = false;
    beamDisplayCard.scrollIntoView({ behavior: "smooth", block: "center" });
    void renderCurrentFrame(true);
  });
  beamPauseButton.addEventListener("click", () => {
    if (!transfer) return;
    if (playing) stopBeam();
    else {
      playing = true;
      beamPauseButton.textContent = "Pause";
      void renderCurrentFrame(true);
    }
  });
  beamFullscreenButton.addEventListener("click", () => void beamScreen.requestFullscreen?.());

  const video = document.querySelector<HTMLVideoElement>("#beam-video")!;
  const scanCanvas = document.querySelector<HTMLCanvasElement>("#beam-scan-canvas")!;
  const scanContext = scanCanvas.getContext("2d", { willReadFrequently: true })!;
  const scannerIdle = document.querySelector<HTMLElement>("#scanner-idle")!;
  const scannerStartButton = document.querySelector<HTMLButtonElement>("#scanner-start-button")!;
  const scannerStopButton = document.querySelector<HTMLButtonElement>("#scanner-stop-button")!;
  const scannerError = document.querySelector<HTMLElement>("#scanner-error")!;
  const receiveTitle = document.querySelector<HTMLElement>("#receive-title")!;
  const receivePercentage = document.querySelector<HTMLElement>("#receive-percentage")!;
  const receiveProgressBar = document.querySelector<HTMLElement>("#receive-progress-bar")!;
  const receiveDetail = document.querySelector<HTMLElement>("#receive-detail")!;
  const receiveFrames = document.querySelector<HTMLElement>("#receive-frames")!;
  const receiveBytes = document.querySelector<HTMLElement>("#receive-bytes")!;
  const receiveVerify = document.querySelector<HTMLElement>("#receive-verify")!;
  const receiveTransportBadge = document.querySelector<HTMLElement>("#receive-transport-badge")!;
  const receivedModelStage = document.querySelector<HTMLElement>("#received-model-stage")!;
  const receivedViewer = document.querySelector<HTMLElement>("#received-viewer")!;
  let receiveBuffer: ReceiveBuffer | null = null;
  let lastScanAt = 0;
  const receiveTransport: BeamTransport = new URL(window.location.href).searchParams.get("beam") === "color3" ? "color3" : "mono";

  if (receiveTransport === "color3") {
    receiveTransportBadge.textContent = "RGB ×3 LAB";
    receiveTransportBadge.classList.add("color3");
    receiveDetail.textContent = "Color receiver armed. Point the camera at an RGB ×3 Spatial Drop beam.";
  }

  function initializeBuffer(frame: ParsedBeamFrame) {
    receiveBuffer = { id: frame.id, hash: frame.hash, name: frame.name, size: frame.size, total: frame.total, chunks: new Map(), finalizing: false };
    receiveTitle.textContent = frame.name;
    receiveDetail.textContent = "Beam found. Keep the transmitting code centered until verification completes.";
    receiveVerify.textContent = "Collecting";
  }

  async function completeTransfer(buffer: ReceiveBuffer) {
    buffer.finalizing = true;
    receiveVerify.textContent = "Checking…";
    receiveDetail.textContent = "All frames received. Verifying SHA-256 integrity…";
    try {
      const bytes = await assembleBeamTransfer(buffer);
      if (receivedObjectUrl) URL.revokeObjectURL(receivedObjectUrl);
      receivedObjectUrl = URL.createObjectURL(new Blob([bytes], { type: "model/gltf-binary" }));
      receivedViewer.setAttribute("src", receivedObjectUrl);
      receivedModelStage.hidden = false;
      receiveVerify.textContent = "Verified";
      receiveDetail.textContent = "Model reconstructed locally. Nothing was uploaded or fetched from model storage.";
      stopBeamScanner();
      window.dispatchEvent(new CustomEvent("spatialdrop:modelreceived", { detail: receivedObjectUrl }));
    } catch (reason) {
      buffer.finalizing = false;
      receiveVerify.textContent = "Retrying";
      receiveDetail.textContent = reason instanceof Error ? reason.message : "Verification failed. Keep scanning another loop.";
    }
  }

  function consumeFrame(value: string) {
    const frame = parseBeamFrame(value);
    if (!frame) return;
    if (!receiveBuffer || receiveBuffer.id !== frame.id) initializeBuffer(frame);
    const buffer = receiveBuffer;
    if (!buffer || buffer.finalizing || buffer.chunks.has(frame.index)) return;
    buffer.chunks.set(frame.index, frame.bytes);
    const count = buffer.chunks.size;
    const percent = Math.min(100, Math.round((count / buffer.total) * 100));
    const byteCount = Array.from(buffer.chunks.values()).reduce((sum, chunk) => sum + chunk.length, 0);
    receivePercentage.textContent = `${percent}%`;
    receiveProgressBar.style.width = `${percent}%`;
    receiveFrames.textContent = `${count} / ${buffer.total}`;
    receiveBytes.textContent = formatBytes(Math.min(byteCount, buffer.size));
    if (count === buffer.total) void completeTransfer(buffer);
  }

  function scanFrame(timestamp: number) {
    if (!scannerStream) return;
    const scanInterval = receiveTransport === "color3" ? 140 : 80;
    if (timestamp - lastScanAt > scanInterval && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
      lastScanAt = timestamp;
      const maxScanWidth = receiveTransport === "color3" ? 640 : 960;
      const sourceSize = Math.min(video.videoWidth, video.videoHeight);
      const sourceX = (video.videoWidth - sourceSize) / 2;
      const sourceY = (video.videoHeight - sourceSize) / 2;
      const targetSize = Math.max(1, Math.round(Math.min(sourceSize, maxScanWidth)));
      scanCanvas.width = targetSize;
      scanCanvas.height = targetSize;
      scanContext.drawImage(video, sourceX, sourceY, sourceSize, sourceSize, 0, 0, targetSize, targetSize);
      const image = scanContext.getImageData(0, 0, scanCanvas.width, scanCanvas.height);
      if (receiveTransport === "color3") {
        const decoded = new Set<string>();
        ([0, 1, 2] as const).forEach((channel) => {
          const code = jsQR(isolateColorChannel(image, channel), image.width, image.height, { inversionAttempts: "dontInvert" });
          if (code?.data.startsWith(`${PROTOCOL}|`) && !decoded.has(code.data)) {
            decoded.add(code.data);
            consumeFrame(code.data);
          }
        });
      } else {
        const code = jsQR(image.data, image.width, image.height, { inversionAttempts: "dontInvert" });
        if (code?.data.startsWith(`${PROTOCOL}|`)) consumeFrame(code.data);
      }
    }
    scannerAnimation = requestAnimationFrame(scanFrame);
  }

  async function startScanner() {
    stopBeamScanner();
    lastScanAt = 0;
    setMessage(scannerError);
    try {
      scannerStream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      video.srcObject = scannerStream;
      await video.play();
      document.querySelector<HTMLElement>(".scanner-viewport")?.classList.add("camera-active");
      scannerIdle.hidden = true;
      scannerStartButton.hidden = true;
      scannerStopButton.hidden = false;
      scannerAnimation = requestAnimationFrame(scanFrame);
    } catch (reason) {
      stopBeamScanner();
      setMessage(scannerError, reason instanceof Error ? reason.message : "Camera access is required to receive a QR Beam.");
    }
  }

  scannerStartButton.addEventListener("click", () => void startScanner());
  scannerStopButton.addEventListener("click", stopBeamScanner);
  window.addEventListener("spatialdrop:modechange", (event) => {
    if ((event as CustomEvent<string>).detail !== "beam") stopBeam();
  });
  window.addEventListener("pagehide", () => {
    stopBeam();
    stopBeamScanner();
    if (receivedObjectUrl) URL.revokeObjectURL(receivedObjectUrl);
  });

  updateTransportUi();
}
