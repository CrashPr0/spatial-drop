import QRCode from "qrcode";
import jsQR from "jsqr";
import {
  BEAM_CHUNK_BYTES,
  BEAM_PROTOCOL,
  type BeamTransfer,
  type ParsedBeamFrame,
  type ReceiveBuffer,
  addBeamFrame,
  assembleBeamTransfer,
  createBeamFrame,
  createBeamTransfer,
  createCimbarFile,
  createReceiveBuffer,
  parseBeamFrame,
  verifyCimbarModel,
} from "./beam-codec";

const FRAME_DELAY_MS = 330;
const QR_RENDER_WIDTH = 720;
const QR_VERSION = 18;
const CIMBAR_MODE = "Bu";
const CIMBAR_FPS = 10;
type BeamTransport = "mono" | "cimbar";

let scannerStream: MediaStream | null = null;
let scannerAnimation = 0;
let receivedObjectUrl = "";

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  return `${(bytes / 1024).toFixed(1)} KB`;
}

function setMessage(element: HTMLElement, message = "") {
  element.textContent = message;
  element.hidden = !message;
}

function cimbarUrl(file: "index.html" | "recv.html") {
  return new URL(`cimbar/${file}?embed=1`, `${window.location.origin}${window.location.pathname}`).toString();
}

export function stopBeamScanner() {
  cancelAnimationFrame(scannerAnimation);
  scannerAnimation = 0;
  scannerStream?.getTracks().forEach((track) => track.stop());
  scannerStream = null;
  const video = document.querySelector<HTMLVideoElement>("#beam-video");
  const scanCanvas = document.querySelector<HTMLCanvasElement>("#beam-scan-canvas");
  const viewport = document.querySelector<HTMLElement>(".scanner-viewport");
  const cimbarFrame = document.querySelector<HTMLIFrameElement>("#cimbar-receiver-frame");
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
  if (cimbarFrame) {
    cimbarFrame.contentWindow?.postMessage({ type: "spatialdrop:cimbar-stop" }, window.location.origin);
    cimbarFrame.hidden = true;
    cimbarFrame.src = "about:blank";
  }
  viewport?.classList.remove("camera-active", "cimbar-active");
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
  const cimbarSenderFrame = document.querySelector<HTMLIFrameElement>("#cimbar-sender-frame")!;
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
  let sequence = 0;
  let playing = false;
  let cimbarReady = false;
  let transport: BeamTransport = new URL(window.location.href).searchParams.get("beam") === "cimbar" ? "cimbar" : "mono";

  async function renderReceiverLink() {
    const receiverUrl = new URL(window.location.href);
    receiverUrl.search = "";
    if (transport === "cimbar") receiverUrl.searchParams.set("beam", "cimbar");
    receiverUrl.hash = "receive";
    await QRCode.toCanvas(receiverLinkCanvas, receiverUrl.toString(), {
      width: 420,
      margin: 3,
      errorCorrectionLevel: "H",
      color: { dark: "#0055A2", light: "#ffffff" },
    });
  }

  function stopCimbarSender() {
    if (!cimbarSenderFrame.hidden) {
      cimbarSenderFrame.contentWindow?.postMessage({ type: "spatialdrop:cimbar-pause", paused: true }, window.location.origin);
      cimbarSenderFrame.hidden = true;
      cimbarSenderFrame.src = "about:blank";
    }
    cimbarReady = false;
  }

  function stopBeam() {
    playing = false;
    window.clearTimeout(beamTimer);
    beamPauseButton.textContent = "Resume";
    if (transport === "cimbar") cimbarSenderFrame.contentWindow?.postMessage({ type: "spatialdrop:cimbar-pause", paused: true }, window.location.origin);
  }

  function updateTransferSummary() {
    if (!transfer) return;
    if (transport === "cimbar") {
      beamFileDetail.textContent = `${formatBytes(transfer.size)} · Cimbar Mode ${CIMBAR_MODE}`;
      beamTransferSummary.textContent = `${transfer.name} · ${formatBytes(transfer.size)} · maximum-compatibility symbols, zstd compression, Reed–Solomon correction, and a ${CIMBAR_FPS} FPS fountain stream.`;
    } else {
      const ratio = Math.round((transfer.transportSize / transfer.size) * 100);
      const compression = transfer.compressed ? `${formatBytes(transfer.transportSize)} after gzip (${ratio}%)` : "raw payload";
      beamFileDetail.textContent = `${formatBytes(transfer.size)} · ${compression} · ${transfer.fragments.length} source fragments`;
      beamTransferSummary.textContent = `${transfer.name} · ${transfer.fragments.length} source fragments · compact Base32 QR version ${QR_VERSION}. After the source pass, every new frame adds rateless recovery data.`;
    }
  }

  function updateTransportUi() {
    transportInputs.forEach((input) => { input.checked = input.value === transport; });
    const isCimbar = transport === "cimbar";
    transportNote.textContent = isCimbar
      ? "Camera-friendly profile: resilient Bu symbols at 10 FPS, with the receiver locked to the same mode for more decode attempts."
      : "Reliable fallback: compressed Base32 frames plus fountain recovery. No exact missed frame is required.";
    transportSpec.textContent = isCimbar ? "Cimbar WASM" : "Fountain QR";
    chunkSpec.textContent = isCimbar ? "zstd + Wirehair" : `${BEAM_CHUNK_BYTES} B + parity`;
    beamScreen.classList.toggle("cimbar", isCimbar);
    updateTransferSummary();
    void renderReceiverLink();
  }

  async function renderCurrentFrame(scheduleNext = false) {
    if (!transfer || transport !== "mono") return;
    try {
      const value = createBeamFrame(transfer, sequence);
      await QRCode.toCanvas(beamCanvas, value, {
        width: QR_RENDER_WIDTH,
        margin: 4,
        version: QR_VERSION,
        errorCorrectionLevel: "L",
        color: { dark: "#001f3f", light: "#ffffff" },
      });
      if (sequence < transfer.fragments.length) {
        beamFrameLabel.textContent = `SOURCE ${sequence + 1} / ${transfer.fragments.length}`;
      } else {
        beamFrameLabel.textContent = `RECOVERY +${sequence - transfer.fragments.length + 1}`;
      }
      beamLoopLabel.textContent = "RATELESS STREAM";
      if (!scheduleNext || !playing) return;
      sequence = sequence >= 0xfffffffe ? transfer.fragments.length : sequence + 1;
      beamTimer = window.setTimeout(() => void renderCurrentFrame(true), FRAME_DELAY_MS);
    } catch (reason) {
      stopBeam();
      setMessage(beamError, reason instanceof Error ? reason.message : "Could not render the optical stream.");
    }
  }

  function sendCimbarModel() {
    if (!transfer || !cimbarReady || transport !== "cimbar") return;
    cimbarSenderFrame.contentWindow?.postMessage({
      type: "spatialdrop:cimbar-send",
      file: createCimbarFile(transfer),
      mode: CIMBAR_MODE,
      fps: CIMBAR_FPS,
    }, window.location.origin);
    beamFrameLabel.textContent = `CIMBAR MODE ${CIMBAR_MODE} · ${CIMBAR_FPS} FPS`;
    beamLoopLabel.textContent = "FOUNTAIN STREAM";
  }

  function startCimbarSender() {
    if (!transfer) return;
    window.clearTimeout(beamTimer);
    beamCanvas.hidden = true;
    cimbarSenderFrame.hidden = false;
    cimbarReady = false;
    cimbarSenderFrame.src = cimbarUrl("index.html");
    beamFrameLabel.textContent = "STARTING CIMBAR WASM…";
    beamLoopLabel.textContent = "TURBO";
  }

  async function acceptBeamFile(file?: File) {
    if (!file) return;
    stopBeam();
    stopCimbarSender();
    setMessage(beamError);
    beamStartButton.disabled = true;
    beamFileTitle.textContent = file.name;
    beamFileDetail.textContent = "Compressing, hashing, and building recovery fragments…";
    try {
      transfer = await createBeamTransfer(file);
      sequence = 0;
      updateTransferSummary();
      beamStartButton.disabled = false;
      beamStartButton.querySelector("span")!.textContent = transport === "cimbar" ? "Start Cimbar Reliable" : "Start fountain beam";
      if (transport === "mono") {
        beamCanvas.hidden = false;
        await renderCurrentFrame();
      }
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
    stopCimbarSender();
    transport = input.value === "cimbar" ? "cimbar" : "mono";
    sequence = 0;
    beamCanvas.hidden = transport === "cimbar";
    beamStartButton.querySelector("span")!.textContent = transport === "cimbar" ? "Start Cimbar Reliable" : "Start fountain beam";
    updateTransportUi();
    if (transfer && transport === "mono") void renderCurrentFrame();
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
    if (transport === "cimbar") startCimbarSender();
    else {
      stopCimbarSender();
      beamCanvas.hidden = false;
      void renderCurrentFrame(true);
    }
  });
  beamPauseButton.addEventListener("click", () => {
    if (!transfer) return;
    if (playing) {
      stopBeam();
    } else {
      playing = true;
      beamPauseButton.textContent = "Pause";
      if (transport === "cimbar") cimbarSenderFrame.contentWindow?.postMessage({ type: "spatialdrop:cimbar-pause", paused: false }, window.location.origin);
      else void renderCurrentFrame(true);
    }
  });
  beamFullscreenButton.addEventListener("click", () => void beamScreen.requestFullscreen?.());

  const video = document.querySelector<HTMLVideoElement>("#beam-video")!;
  const scanCanvas = document.querySelector<HTMLCanvasElement>("#beam-scan-canvas")!;
  const scanContext = scanCanvas.getContext("2d", { willReadFrequently: true })!;
  const scannerViewport = document.querySelector<HTMLElement>(".scanner-viewport")!;
  const scannerIdle = document.querySelector<HTMLElement>("#scanner-idle")!;
  const scannerStartButton = document.querySelector<HTMLButtonElement>("#scanner-start-button")!;
  const scannerStopButton = document.querySelector<HTMLButtonElement>("#scanner-stop-button")!;
  const scannerError = document.querySelector<HTMLElement>("#scanner-error")!;
  const cimbarReceiverFrame = document.querySelector<HTMLIFrameElement>("#cimbar-receiver-frame")!;
  const receiverTransportButtons = Array.from(document.querySelectorAll<HTMLButtonElement>("[data-receive-transport]"));
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
  let receiveTransport: BeamTransport = new URL(window.location.href).searchParams.get("beam") === "cimbar" ? "cimbar" : "mono";

  function resetReceiveProgress() {
    receiveBuffer = null;
    receiveTitle.textContent = "Waiting for beam";
    receivePercentage.textContent = "0%";
    receiveProgressBar.style.width = "0%";
    receiveFrames.textContent = "0 / —";
    receiveBytes.textContent = "0 KB";
    receiveVerify.textContent = "Waiting";
  }

  function updateReceiveTransportUi(updateUrl = false) {
    const isCimbar = receiveTransport === "cimbar";
    receiverTransportButtons.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.receiveTransport === receiveTransport)));
    receiveTransportBadge.textContent = isCimbar ? "CIMBAR Bu · 10 FPS" : "FOUNTAIN QR";
    receiveTransportBadge.classList.toggle("cimbar", isCimbar);
    scannerStartButton.querySelector("span")!.textContent = isCimbar ? "Start Cimbar receiver" : "Start Fountain QR scanner";
    receiveDetail.textContent = isCimbar
      ? "Cimbar receiver armed. It accepts fountain frames in any order and corrects damaged symbols."
      : "Open QR Beam on another screen, choose a compact GLB, and start transmitting.";
    if (updateUrl) {
      const url = new URL(window.location.href);
      if (isCimbar) url.searchParams.set("beam", "cimbar");
      else url.searchParams.delete("beam");
      window.history.replaceState(null, "", url);
    }
  }

  function initializeBuffer(frame: ParsedBeamFrame) {
    receiveBuffer = createReceiveBuffer(frame);
    receiveTitle.textContent = "Spatial Drop GLB";
    receiveDetail.textContent = "Beam found. Every independent frame raises the recovery rank; exact missed frames are not required.";
    receiveVerify.textContent = "Collecting";
  }

  function publishReceivedModel(model: { bytes: Uint8Array; name: string }, method: string) {
    if (receivedObjectUrl) URL.revokeObjectURL(receivedObjectUrl);
    const copy = new Uint8Array(model.bytes.length);
    copy.set(model.bytes);
    receivedObjectUrl = URL.createObjectURL(new Blob([copy.buffer], { type: "model/gltf-binary" }));
    receivedViewer.setAttribute("src", receivedObjectUrl);
    receivedModelStage.hidden = false;
    receiveTitle.textContent = model.name;
    receivePercentage.textContent = "100%";
    receiveProgressBar.style.width = "100%";
    receiveBytes.textContent = formatBytes(model.bytes.length);
    receiveVerify.textContent = "Verified";
    receiveDetail.textContent = `${method} model reconstructed locally and verified with SHA-256. Nothing was uploaded.`;
    stopBeamScanner();
    window.dispatchEvent(new CustomEvent("spatialdrop:modelreceived", { detail: receivedObjectUrl }));
  }

  async function completeTransfer(buffer: ReceiveBuffer) {
    buffer.finalizing = true;
    receiveVerify.textContent = "Checking…";
    receiveDetail.textContent = "Enough independent equations received. Solving, decompressing, and verifying SHA-256…";
    try {
      publishReceivedModel(await assembleBeamTransfer(buffer), "Fountain QR");
    } catch (reason) {
      buffer.finalizing = false;
      receiveVerify.textContent = "Recovering";
      receiveDetail.textContent = reason instanceof Error ? reason.message : "Verification failed. Keep scanning recovery frames.";
    }
  }

  function consumeFrame(value: string) {
    const frame = parseBeamFrame(value);
    if (!frame) return;
    if (!receiveBuffer || receiveBuffer.id !== frame.id) initializeBuffer(frame);
    const buffer = receiveBuffer;
    if (!buffer || buffer.finalizing) return;
    const advanced = addBeamFrame(buffer, frame);
    if (!advanced) return;
    const percent = Math.min(100, Math.round((buffer.rank / buffer.total) * 100));
    receivePercentage.textContent = `${percent}%`;
    receiveProgressBar.style.width = `${percent}%`;
    receiveFrames.textContent = `${buffer.rank} / ${buffer.total} rank`;
    receiveBytes.textContent = formatBytes(Math.min(buffer.rank * buffer.fragmentSize, buffer.messageLength));
    if (buffer.rank === buffer.total) void completeTransfer(buffer);
  }

  function scanFrame(timestamp: number) {
    if (!scannerStream) return;
    if (timestamp - lastScanAt > 80 && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
      lastScanAt = timestamp;
      const sourceSize = Math.min(video.videoWidth, video.videoHeight);
      const sourceX = (video.videoWidth - sourceSize) / 2;
      const sourceY = (video.videoHeight - sourceSize) / 2;
      const targetSize = Math.max(1, Math.round(Math.min(sourceSize, 960)));
      scanCanvas.width = targetSize;
      scanCanvas.height = targetSize;
      scanContext.drawImage(video, sourceX, sourceY, sourceSize, sourceSize, 0, 0, targetSize, targetSize);
      const image = scanContext.getImageData(0, 0, scanCanvas.width, scanCanvas.height);
      const code = jsQR(image.data, image.width, image.height, { inversionAttempts: "dontInvert" });
      if (code?.data.startsWith(`${BEAM_PROTOCOL}:`)) consumeFrame(code.data);
    }
    scannerAnimation = requestAnimationFrame(scanFrame);
  }

  async function startQrScanner() {
    stopBeamScanner();
    lastScanAt = 0;
    setMessage(scannerError);
    try {
      scannerStream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: { ideal: "environment" }, width: { ideal: 1920 }, height: { ideal: 1080 } },
      });
      video.srcObject = scannerStream;
      await video.play();
      scannerViewport.classList.add("camera-active");
      scannerIdle.hidden = true;
      scannerStartButton.hidden = true;
      scannerStopButton.hidden = false;
      scannerAnimation = requestAnimationFrame(scanFrame);
    } catch (reason) {
      stopBeamScanner();
      setMessage(scannerError, reason instanceof Error ? reason.message : "Camera access is required to receive a QR Beam.");
    }
  }

  function startCimbarScanner() {
    stopBeamScanner();
    setMessage(scannerError);
    scannerIdle.hidden = true;
    scannerStartButton.hidden = true;
    scannerStopButton.hidden = false;
    scannerViewport.classList.add("cimbar-active");
    cimbarReceiverFrame.hidden = false;
    cimbarReceiverFrame.src = cimbarUrl("recv.html");
    receiveVerify.textContent = "Starting…";
    receiveDetail.textContent = "Loading the Cimbar WebAssembly camera decoder…";
  }

  scannerStartButton.addEventListener("click", () => {
    if (receiveTransport === "cimbar") startCimbarScanner();
    else void startQrScanner();
  });
  scannerStopButton.addEventListener("click", stopBeamScanner);
  receiverTransportButtons.forEach((button) => button.addEventListener("click", () => {
    const nextTransport: BeamTransport = button.dataset.receiveTransport === "cimbar" ? "cimbar" : "mono";
    if (nextTransport === receiveTransport) return;
    stopBeamScanner();
    receiveTransport = nextTransport;
    resetReceiveProgress();
    updateReceiveTransportUi(true);
  }));

  window.addEventListener("message", (event) => {
    if (event.origin !== window.location.origin || !event.data) return;
    if (event.source === cimbarSenderFrame.contentWindow && event.data.type === "spatialdrop:cimbar-ready") {
      cimbarReady = true;
      sendCimbarModel();
      return;
    }
    if (event.source !== cimbarReceiverFrame.contentWindow) return;
    if (event.data.type === "spatialdrop:cimbar-ready") {
      receiveVerify.textContent = "Scanning";
      receiveDetail.textContent = "Cimbar Bu is ready. Center the complete color field; each frame holds for 100 ms for faster transfer while preserving repeat exposure.";
    } else if (event.data.type === "spatialdrop:cimbar-progress") {
      const values = Array.isArray(event.data.report) ? event.data.report.filter((value: unknown) => typeof value === "number") as number[] : [];
      const progress = values.length ? Math.max(...values) : 0;
      const percent = Math.max(0, Math.min(99, Math.round(progress * 100)));
      receivePercentage.textContent = `${percent}%`;
      receiveProgressBar.style.width = `${percent}%`;
      receiveFrames.textContent = "Fountain stream";
      receiveBytes.textContent = "zstd payload";
      receiveVerify.textContent = "Recovering";
    } else if (event.data.type === "spatialdrop:cimbar-model" && event.data.buffer instanceof ArrayBuffer) {
      receiveVerify.textContent = "Checking…";
      receiveDetail.textContent = "Cimbar fountain complete. Verifying the original GLB…";
      void verifyCimbarModel(String(event.data.name || ""), event.data.buffer)
        .then((model) => publishReceivedModel(model, "Cimbar"))
        .catch((reason) => {
          receiveVerify.textContent = "Failed";
          receiveDetail.textContent = reason instanceof Error ? reason.message : "The Cimbar payload could not be verified.";
        });
    } else if (event.data.type === "spatialdrop:cimbar-error") {
      setMessage(scannerError, String(event.data.message || "The Cimbar decoder stopped unexpectedly."));
    }
  });

  window.addEventListener("spatialdrop:modechange", (event) => {
    if ((event as CustomEvent<string>).detail !== "beam") {
      stopBeam();
      stopCimbarSender();
    }
  });
  window.addEventListener("pagehide", () => {
    stopBeam();
    stopCimbarSender();
    stopBeamScanner();
    if (receivedObjectUrl) URL.revokeObjectURL(receivedObjectUrl);
  });

  updateTransportUi();
  updateReceiveTransportUi();
}
