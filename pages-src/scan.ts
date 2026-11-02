import { BufferAttribute, BufferGeometry, CanvasTexture, LinearFilter, Mesh, MeshStandardMaterial, SRGBColorSpace } from "three";
import { GLTFExporter } from "three/addons/exporters/GLTFExporter.js";
import { cropToSubject } from "./scan-crop";
import { subjectCylinder, type ScanCamera, type Vec3 } from "./scan-mesh";
import { buildUsdz } from "./scan-usdz";
import type { MeshJob, MeshWorkerMessage, ViewMeta } from "./scan-worker";

// the reconstruction library (~1 MB) loads only when someone actually scans
const loadSplat = () => import("./vendor/splat/index.js");
type SplatSession = ReturnType<Awaited<ReturnType<typeof loadSplat>>["createSession"]>;
const isVideoFile = (file: File) => /^video\//.test(file.type) || /\.(mp4|mov|m4v|webm|mkv)$/i.test(file.name);
type ScanState = "idle" | "capturing" | "building" | "training" | "meshing" | "ready";

const MAX_PHOTOS = 72;
const MIN_PHOTOS = 12;
// 35 mm-equivalent focal of a typical phone main camera (iPhone and Pixel wide lenses: 24–26 mm)
const PHONE_MAIN_LENS_MM = 26;
const STAGE_LABELS: Record<string, string> = {
  decode: "Reading photos",
  features: "Finding features",
  matching: "Matching views",
  focal: "Estimating the lens",
  pass: "Solving camera poses",
  register: "Placing cameras",
  ba: "Refining cameras",
  solved: "Cameras solved",
  seed: "Seeding the splat",
  train: "Training on this device",
};
const STAGE_SPANS: Record<string, [number, number]> = {
  decode: [0, 6], features: [6, 18], matching: [18, 30], focal: [30, 34], pass: [30, 34],
  register: [34, 42], ba: [42, 46], solved: [46, 48], seed: [48, 50], train: [50, 96],
};

// triangle budgets per detail level; texture detail is baked separately
const DETAIL_TRIANGLES: Record<string, number> = { light: 6000, standard: 15000, high: 40000 };

type MeshResult = Extract<MeshWorkerMessage, { type: "done" }>;

/** Run one meshing job in a fresh worker; `signal` aborts it by terminating the worker. */
function runMeshWorker(job: MeshJob, onProgress: (message: Extract<MeshWorkerMessage, { type: "progress" }>) => void, signal: AbortSignal) {
  return new Promise<MeshResult>((resolve, reject) => {
    const worker = new Worker(new URL("./scan-worker.ts", import.meta.url), { type: "module" });
    const finish = () => { worker.terminate(); signal.removeEventListener("abort", abort); };
    const abort = () => { finish(); reject(new DOMException("Superseded", "AbortError")); };
    signal.addEventListener("abort", abort);
    worker.onmessage = ({ data }: MessageEvent<MeshWorkerMessage>) => {
      if (data.type === "progress") return onProgress(data);
      finish();
      if (data.type === "done") resolve(data);
      else reject(new Error(data.message));
    };
    worker.onerror = (event) => { finish(); reject(new Error(event.message || "The meshing worker failed.")); };
    worker.postMessage(job);
  });
}

/** Solver cameras (world-to-camera R, t; OpenCV axes) as the mesher's orbit cameras. */
function orbitCameras(cams: { R: number[]; t: number[] }[]): ScanCamera[] {
  return cams.map(({ R, t }) => ({
    position: [
      -(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]),
      -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]),
      -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2]),
    ] as Vec3,
    forward: [R[6], R[7], R[8]] as Vec3,
    up: [-R[3], -R[4], -R[5]] as Vec3,
  }));
}

const isPhone = () => matchMedia("(any-pointer: coarse)").matches && Math.min(screen.width, screen.height) <= 820;

function formatBytes(bytes: number) {
  return bytes < 1024 * 1024 ? `${Math.ceil(bytes / 1024)} KB` : `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function download(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 30000);
}

/** Cheap focus score: variance of the Laplacian on a small grayscale copy. */
function sharpness(gray: Float32Array, width: number, height: number) {
  let sum = 0, sumSq = 0, count = 0;
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const i = y * width + x;
      const lap = 4 * gray[i] - gray[i - 1] - gray[i + 1] - gray[i - width] - gray[i + width];
      sum += lap; sumSq += lap * lap; count++;
    }
  }
  const mean = sum / count;
  return sumSq / count - mean * mean;
}

export function setupScanLab() {
  const $ = <T extends HTMLElement>(selector: string) => document.querySelector<T>(selector)!;
  const video = $<HTMLVideoElement>("#scan-video");
  const viewport = $(".scan-viewport");
  const idle = $("#scan-idle");
  const cameraButton = $<HTMLButtonElement>("#scan-camera-button");
  const stopButton = $<HTMLButtonElement>("#scan-stop-button");
  const fileInput = $<HTMLInputElement>("#scan-file-input");
  const dropZone = $("#scan-drop-zone");
  const thumbs = $("#scan-thumbs");
  const photoCount = $("#scan-photo-count");
  const captureHint = $("#scan-capture-hint");
  const clearButton = $<HTMLButtonElement>("#scan-clear-button");
  const buildButton = $<HTMLButtonElement>("#scan-build-button");
  const error = $("#scan-error");
  const stageTitle = $("#scan-stage-title");
  const percentage = $("#scan-percentage");
  const progressBar = $("#scan-progress-bar");
  const detail = $("#scan-detail");
  const metricCameras = $("#scan-metric-cameras");
  const metricSplats = $("#scan-metric-splats");
  const metricTime = $("#scan-metric-time");
  const liveCanvas = $<HTMLCanvasElement>("#scan-live-canvas");
  const finishButton = $<HTMLButtonElement>("#scan-finish-button");
  const meshPanel = $("#scan-mesh-panel");
  const heightInput = $<HTMLInputElement>("#scan-height");
  const cropInput = $<HTMLInputElement>("#scan-crop");
  const densityInput = $<HTMLInputElement>("#scan-density");
  const detailInput = $<HTMLSelectElement>("#scan-detail-level");
  const rebuildButton = $<HTMLButtonElement>("#scan-rebuild-button");
  const viewer = $<HTMLElement & { src: string }>("#scan-viewer");
  const meshStats = $("#scan-mesh-stats");
  const studioButton = $<HTMLButtonElement>("#scan-studio-button");
  const glbButton = $<HTMLButtonElement>("#scan-glb-button");
  const plyButton = $<HTMLButtonElement>("#scan-ply-button");
  const usdzButton = $<HTMLButtonElement>("#scan-usdz-button");
  const arButton = $<HTMLButtonElement>("#scan-ar-button");

  let state: ScanState = "idle";
  let photos: File[] = [];
  let stream: MediaStream | null = null;
  let captureTimer = 0;
  let session: SplatSession | null = null;
  let splat: { data: Float32Array; n: number; dc: "sigmoid" | "sh" } | null = null;
  let cameras: ScanCamera[] = [];
  let viewMetas: ViewMeta[] = [];
  let fullIntrinsics = new Map<number, Pick<ViewMeta, "featureWidth" | "featureHeight" | "cx" | "cy">>();
  let scannedPhotos: File[] = [];
  let glbBlob: Blob | null = null;
  let glbUrl = "";
  let usdzBlob: Blob | null = null;
  let usdzUrl = "";
  let startedAt = 0;
  let clockTimer = 0;
  let lookTimer = 0;

  function setError(message = "") {
    error.textContent = message;
    error.hidden = !message;
  }

  function setProgress(title: string, value: number, text?: string) {
    stageTitle.textContent = title;
    const clamped = Math.max(0, Math.min(100, Math.round(value)));
    percentage.textContent = `${clamped}%`;
    progressBar.style.width = `${clamped}%`;
    if (text !== undefined) detail.textContent = text;
  }

  function setState(next: ScanState) {
    state = next;
    const busy = next === "building" || next === "training" || next === "meshing";
    buildButton.disabled = busy || photos.length < MIN_PHOTOS;
    cameraButton.disabled = busy;
    clearButton.disabled = busy || !photos.length;
    dropZone.classList.toggle("disabled", busy);
    finishButton.hidden = next !== "training";
    rebuildButton.disabled = next !== "ready";
  }

  function renderPhotos() {
    photoCount.textContent = `${photos.length}`;
    const needed = Math.max(0, MIN_PHOTOS - photos.length);
    captureHint.textContent = photos.length >= MAX_PHOTOS
      ? "Photo limit reached. Build when ready."
      : needed
        ? `${needed} more to start · 30–60 recommended`
        : photos.length < 30 ? "Enough to try · more angles help" : "Great coverage";
    thumbs.replaceChildren(...photos.slice(-12).map((photo) => {
      const image = document.createElement("img");
      image.alt = "";
      image.src = URL.createObjectURL(photo);
      image.addEventListener("load", () => URL.revokeObjectURL(image.src), { once: true });
      return image;
    }));
    setState(state);
  }

  function addPhoto(file: File) {
    if (photos.length >= MAX_PHOTOS) return;
    photos.push(file);
    renderPhotos();
  }

  // ── capture ────────────────────────────────────────────────────────────
  function stopCamera() {
    window.clearInterval(captureTimer);
    captureTimer = 0;
    stream?.getTracks().forEach((track) => track.stop());
    stream = null;
    video.srcObject = null;
    viewport.classList.remove("camera-active");
    idle.hidden = false;
    cameraButton.hidden = false;
    stopButton.hidden = true;
    if (state === "capturing") setState("idle");
  }

  async function startCamera() {
    setError();
    if (!navigator.mediaDevices?.getUserMedia) return setError("This browser cannot open the camera. Choose photos or a video instead.");
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" }, width: { ideal: 1920 }, height: { ideal: 1080 } },
        audio: false,
      });
    } catch {
      return setError("Camera access was blocked. Allow the camera, or choose photos or a video instead.");
    }
    video.srcObject = stream;
    await video.play().catch(() => undefined);
    viewport.classList.add("camera-active");
    idle.hidden = true;
    cameraButton.hidden = true;
    stopButton.hidden = false;
    setState("capturing");

    // keep a frame when the view has moved enough and is not blurrier than usual
    const probe = document.createElement("canvas");
    probe.width = 160;
    probe.height = 120;
    const probeContext = probe.getContext("2d", { willReadFrequently: true })!;
    let lastKept: Float32Array | null = null;
    let lastKeptAt = 0;
    const recentSharpness: number[] = [];
    let saving = false;
    captureTimer = window.setInterval(() => {
      if (saving || !video.videoWidth || photos.length >= MAX_PHOTOS) return;
      probeContext.drawImage(video, 0, 0, probe.width, probe.height);
      const rgba = probeContext.getImageData(0, 0, probe.width, probe.height).data;
      const gray = new Float32Array(probe.width * probe.height);
      for (let i = 0; i < gray.length; i++) gray[i] = 0.299 * rgba[i * 4] + 0.587 * rgba[i * 4 + 1] + 0.114 * rgba[i * 4 + 2];
      const focus = sharpness(gray, probe.width, probe.height);
      recentSharpness.push(focus);
      if (recentSharpness.length > 30) recentSharpness.shift();
      const typical = [...recentSharpness].sort((a, b) => a - b)[recentSharpness.length >> 1];
      let motion = Infinity;
      if (lastKept) {
        motion = 0;
        for (let i = 0; i < gray.length; i++) motion += Math.abs(gray[i] - lastKept[i]);
        motion /= gray.length;
      }
      const now = performance.now();
      if (now - lastKeptAt < 350 || motion < 9 || focus < 0.7 * typical) return;
      lastKept = gray;
      lastKeptAt = now;
      saving = true;
      const frame = document.createElement("canvas");
      frame.width = video.videoWidth;
      frame.height = video.videoHeight;
      frame.getContext("2d")!.drawImage(video, 0, 0);
      frame.toBlob((blob) => {
        saving = false;
        if (!blob) return;
        viewport.classList.remove("flash");
        void viewport.offsetWidth;
        viewport.classList.add("flash");
        addPhoto(new File([blob], `capture_${String(photos.length).padStart(3, "0")}.jpg`, { type: "image/jpeg" }));
        if (photos.length >= MAX_PHOTOS) stopCamera();
      }, "image/jpeg", 0.92);
    }, 120);
  }

  async function acceptFiles(files: File[]) {
    setError();
    if (!files.length || state === "building" || state === "training" || state === "meshing") return;
    const clip = files.find((file) => isVideoFile(file));
    if (clip) {
      setState("building");
      try {
        setProgress("Picking sharp frames", 0, `Reading ${clip.name} (${formatBytes(clip.size)}) on this device…`);
        const { extractSharpFrames } = await loadSplat();
        const result = await extractSharpFrames(clip, {
          maxFrames: MAX_PHOTOS,
          onProgress: (event: { stage: string; done: number; total: number }) => {
            const share = event.stage === "scan" ? 0.6 : 0.4;
            const base = event.stage === "scan" ? 0 : 60;
            setProgress("Picking sharp frames", base + share * 100 * (event.done / Math.max(1, event.total)));
          },
        });
        photos = result.frames.slice(0, MAX_PHOTOS).map((frame: { source: Blob; name: string }) =>
          new File([frame.source], `${frame.name}.jpg`, { type: frame.source.type || "image/jpeg" }));
        setProgress("Frames ready", 100, `${photos.length} sharp frames from a ${result.duration.toFixed(1)} s clip.`);
      } catch (reason) {
        setError(reason instanceof Error ? `That video could not be read: ${reason.message}` : "That video could not be read.");
      }
      setState("idle");
      renderPhotos();
      return;
    }
    files.filter((file) => file.type.startsWith("image/") || /\.(jpe?g|png|heic|webp)$/i.test(file.name)).forEach(addPhoto);
  }

  // ── reconstruction ─────────────────────────────────────────────────────
  function stopClock() {
    window.clearInterval(clockTimer);
    window.clearInterval(lookTimer);
  }

  async function build() {
    setError();
    if (!("gpu" in navigator)) {
      return setError("On-device scanning needs WebGPU: Safari 26 or newer on iPhone, or a recent Chrome or Edge.");
    }
    stopCamera();
    session?.dispose();
    session = null;
    splat = null;
    meshPanel.hidden = true;
    liveCanvas.hidden = true;
    setState("building");
    startedAt = performance.now();
    clockTimer = window.setInterval(() => {
      const seconds = Math.round((performance.now() - startedAt) / 1000);
      metricTime.textContent = `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
    }, 1000);

    const { createSession, focalPxFrom35, solveTierOpts } = await loadSplat();
    const phone = isPhone();
    // The splat only has to get the shape right (color comes from the photos), and
    // that settles early. Measured on the synthetic orbit, three runs each:
    // 2000 steps from 20k splats: 8.5-9.1 s, 2.7-3.1 mm, 25.1-25.2 dB held out;
    // 7000 steps from the 60k default: 60-68 s, 3.0-4.2 mm, 24.1-24.3 dB.
    const maxIters = 2000;
    const created = createSession({
      maxIters,
      holdout: -1,
      evalHoldEvery: 1e9,
      initTarget: 20000,
      ...(phone ? { lowMem: true } : {}),
      sfm: { ...solveTierOpts("quick"), uiYield: true, ...(phone ? { workers: 3 } : {}) },
      trainer: { maxSplats: phone ? 300000 : 600000, capMult: 8 },
      // the browser harness (tests/browser/scan-orbit.js) tries other settings through this
      ...(import.meta.env.DEV ? (window as { spatialDropSessionOverrides?: object }).spatialDropSessionOverrides : {}),
    });
    session = created;
    created.on("stage", (event: { stage: string; done: number; total: number }) => {
      const span = STAGE_SPANS[event.stage];
      if (!span || event.stage === "train") return;
      const fraction = event.total ? event.done / event.total : 0;
      setProgress(STAGE_LABELS[event.stage] || "Working", span[0] + (span[1] - span[0]) * fraction,
        event.stage === "register" ? `${event.done} of ${event.total} photos placed in 3D.` : undefined);
      if (event.stage === "register") metricCameras.textContent = `${event.done} / ${event.total}`;
    });
    created.on("metrics", (metrics: { iter: number; splats: number; itersPerSec: number }) => {
      const fraction = metrics.iter / maxIters;
      // early steps run on few splats and are fast: only estimate once growth has settled
      const remaining = metrics.itersPerSec > 0 && fraction > 0.15 ? Math.max(0, (maxIters - metrics.iter) / metrics.itersPerSec) : 0;
      setProgress(STAGE_LABELS.train, 50 + 46 * fraction,
        `${metrics.iter.toLocaleString()} of ${maxIters.toLocaleString()} steps${remaining ? ` · ${remaining < 60 ? "under a minute" : `about ${Math.ceil(remaining / 60)} min`} left` : ""}. You can finish early.`);
      metricSplats.textContent = metrics.splats.toLocaleString();
    });
    created.on("event", (event: { kind: string }) => {
      if (event.kind === "train-complete") void afterTraining();
      if (event.kind === "device-lost") {
        stopClock();
        setState("idle");
        setError("The GPU was reclaimed (the tab went to the background). Keep this tab open while it trains, then build again.");
      }
    });

    try {
      setProgress(STAGE_LABELS.decode, 0, `Processing ${photos.length} photos. Nothing leaves this device.`);
      scannedPhotos = photos.slice();
      await created.load(scannedPhotos);
      // live captures carry no EXIF lens data; without a starting focal the
      // solver's search can lock onto a wrong lens and scatter the cameras
      const frames = created.frames as { fw: number; fh: number; exif?: { f35?: number } | null }[];
      const tagged = frames.filter((frame) => (frame.exif?.f35 ?? 0) > 0).length;
      await created.solve(tagged >= 0.6 * frames.length ? {} : {
        focalPrior: focalPxFrom35(PHONE_MAIN_LENS_MM, frames[0].fw, frames[0].fh),
      });
      const placed = created.recon.cams.length;
      metricCameras.textContent = `${placed} / ${photos.length}`;
      if (placed < Math.max(6, 0.4 * photos.length)) {
        throw new Error(`Only ${placed} of ${photos.length} photos could be placed. Rescan with more overlap: move slowly around the object.`);
      }
      setProgress(STAGE_LABELS.seed, 48, `${placed} cameras solved. Starting the splat trainer…`);
      // texturing and carving project into the full photos: keep their intrinsics
      fullIntrinsics = new Map(created.recon.cams.map((camera: { imgIdx: number; cx?: number; cy?: number }) => {
        const frame = created.frames[camera.imgIdx];
        return [camera.imgIdx, { featureWidth: frame.fw, featureHeight: frame.fh, cx: camera.cx, cy: camera.cy }];
      }));
      // train on the object, not the room: crop every photo to the subject's region
      const region = subjectCylinder(
        created.recon.points.map((point: { X: number[] }) => point.X as Vec3),
        orbitCameras(created.recon.cams),
      );
      cropToSubject(created.frames, created.recon.cams, region);
      await created.seed();
      if (session !== created) return;
      const first = created.trainer.camMeta[0];
      liveCanvas.width = first.w;
      liveCanvas.height = first.h;
      liveCanvas.hidden = false;
      created.view.attach(liveCanvas);
      let look = 0;
      created.view.lookThrough(0);
      lookTimer = window.setInterval(() => {
        look = (look + 1) % created.trainer.camMeta.length;
        // cropped training views differ in size
        const meta = created.trainer.camMeta[look];
        liveCanvas.width = meta.w;
        liveCanvas.height = meta.h;
        created.view.lookThrough(look);
      }, 2500);
      setState("training");
      created.start();
    } catch (reason) {
      stopClock();
      created.dispose();
      if (session === created) session = null;
      setState("idle");
      setProgress("Scan stopped", 0, "");
      setError(reason instanceof Error ? reason.message : "The scan could not be reconstructed.");
    }
  }

  async function afterTraining() {
    const current = session;
    if (!current || state !== "training") return;
    window.clearInterval(lookTimer);
    setState("meshing");
    setProgress("Saving the splat", 97, "Reading the trained splat back from the GPU…");
    try {
      const raw = await current.exportRawState();
      splat = { data: Float32Array.from(raw.data.subarray(0, raw.n * 16)), n: raw.n, dc: current.trainer.dcMode };
      cameras = orbitCameras(current.recon.cams);
      const recon = current.recon;
      viewMetas = recon.cams.map((camera: { imgIdx: number; R: number[]; t: number[]; f: number; fy?: number; cx?: number; cy?: number }) => {
        const frame = current.frames[camera.imgIdx];
        // the uncropped camera, as the photos are
        const full = fullIntrinsics.get(camera.imgIdx) ?? { featureWidth: frame.fw, featureHeight: frame.fh, cx: camera.cx, cy: camera.cy };
        return {
          name: frame.name, R: Array.from(camera.R), t: Array.from(camera.t), f: camera.f, fy: camera.fy,
          k1: recon.k1 ?? 0, k2: recon.k2 ?? 0, ...full,
        };
      });
      current.pause();
      if (import.meta.env.DEV) Object.assign(window, { spatialDropScan: { splat, cameras, viewMetas, session: current } });
      await makeMesh();
    } catch (reason) {
      setState("idle");
      setError(reason instanceof Error ? reason.message : "The trained splat could not be read.");
    } finally {
      stopClock();
    }
  }

  let meshJob: AbortController | null = null;

  async function makeMesh() {
    if (!splat) return;
    meshJob?.abort();
    const job = new AbortController();
    meshJob = job;
    setState("meshing");
    setProgress("Building the AR mesh", 98, "Carving a solid surface out of the splat…");
    try {
      const heightCm = Math.min(500, Math.max(1, Number(heightInput.value) || 30));
      const { mesh, atlas } = await runMeshWorker({
        // a copy: the worker's buffers are gone once it finishes, and rebuilds need the splat again
        splat: { data: splat.data.slice(), n: splat.n, dc: splat.dc },
        cameras,
        viewMetas,
        photos: scannedPhotos,
        options: {
          heightMeters: heightCm / 100,
          crop: Number(cropInput.value),
          density: Number(densityInput.value),
          resolution: 104,
          targetTriangles: DETAIL_TRIANGLES[detailInput.value] ?? DETAIL_TRIANGLES.standard,
        },
        textureSize: 1024,
      }, (progress) => {
        if (progress.stage === "texture") {
          const { done, total } = progress;
          setProgress("Painting the texture", 98.5 + done / total, `Projecting photo ${Math.min(done + 1, total)} of ${total} onto the surface…`);
        }
      }, job.signal);
      const exports = await exportModels(mesh, atlas);
      if (meshJob !== job) return;
      glbBlob = exports.glb;
      usdzBlob = exports.usdz;
      if (glbUrl) URL.revokeObjectURL(glbUrl);
      if (usdzUrl) URL.revokeObjectURL(usdzUrl);
      glbUrl = URL.createObjectURL(glbBlob);
      usdzUrl = URL.createObjectURL(usdzBlob);
      // a ready-made USDZ for Quick Look, so the iPhone does not convert at tap time
      viewer.setAttribute("ios-src", usdzUrl);
      viewer.setAttribute("src", glbUrl);
      const [w, h, d] = mesh.stats.sizeMeters.map((value) => Math.round(value * 100));
      meshStats.textContent = `${mesh.stats.triangles.toLocaleString()} triangles · ${w} × ${h} × ${d} cm · GLB ${formatBytes(glbBlob.size)} · USDZ ${formatBytes(usdzBlob.size)}${mesh.stats.groundRemoved ? " · ground removed" : ""}`;
      meshPanel.hidden = false;
      setProgress("Model ready", 100, "Your scan is a local GLB now. Place it in AR, open it in Studio, or download it.");
      window.dispatchEvent(new CustomEvent("spatialdrop:scanready", { detail: glbUrl }));
      setState("ready");
    } catch (reason) {
      if (reason instanceof DOMException && reason.name === "AbortError") return;
      setState("ready");
      meshPanel.hidden = false;
      setError(reason instanceof Error ? reason.message : "The mesh could not be built.");
    } finally {
      if (meshJob === job) meshJob = null;
    }
  }

  async function exportModels(scan: MeshResult["mesh"], atlas: MeshResult["atlas"]) {
    const canvas = document.createElement("canvas");
    canvas.width = atlas.size;
    canvas.height = atlas.size;
    canvas.getContext("2d")!.putImageData(new ImageData(atlas.pixels, atlas.size, atlas.size), 0, 0);
    const texture = new CanvasTexture(canvas);
    texture.flipY = false;
    texture.colorSpace = SRGBColorSpace;
    texture.generateMipmaps = false;
    texture.minFilter = LinearFilter;
    texture.magFilter = LinearFilter;
    // the atlas is laid out in surface order, so JPEG blocks only blend neighboring surface
    texture.userData.mimeType = "image/jpeg";

    const geometry = new BufferGeometry();
    geometry.setAttribute("position", new BufferAttribute(atlas.positions, 3));
    geometry.setAttribute("normal", new BufferAttribute(atlas.normals, 3));
    geometry.setAttribute("uv", new BufferAttribute(atlas.uvs, 2));
    const material = new MeshStandardMaterial({ map: texture, roughness: 0.85, metalness: 0 });
    const mesh = new Mesh(geometry, material);
    mesh.name = "Spatial Drop scan";
    const [glb, jpeg] = await Promise.all([
      new GLTFExporter().parseAsync(mesh, { binary: true }),
      new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.92)),
    ]);
    if (!(glb instanceof ArrayBuffer)) throw new Error("The GLB export failed.");
    if (!jpeg) throw new Error("The texture could not be encoded.");
    // USD shares the mesh's points and gives each corner its atlas coordinate
    const usdz = buildUsdz({ positions: scan.positions, normals: scan.normals, indices: atlas.indices, uvs: atlas.uvs },
      new Uint8Array(await jpeg.arrayBuffer()));
    return {
      glb: new Blob([glb], { type: "model/gltf-binary" }),
      usdz: new Blob([usdz], { type: "model/vnd.usdz+zip" }),
    };
  }

  // ── wiring ─────────────────────────────────────────────────────────────
  cameraButton.addEventListener("click", () => void startCamera());
  stopButton.addEventListener("click", stopCamera);
  dropZone.addEventListener("click", () => fileInput.click());
  dropZone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") fileInput.click();
  });
  dropZone.addEventListener("dragover", (event) => { event.preventDefault(); dropZone.classList.add("dragging"); });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragging"));
  dropZone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragging");
    void acceptFiles(Array.from(event.dataTransfer?.files || []));
  });
  fileInput.addEventListener("change", () => {
    void acceptFiles(Array.from(fileInput.files || []));
    fileInput.value = "";
  });
  clearButton.addEventListener("click", () => {
    photos = [];
    renderPhotos();
  });
  buildButton.addEventListener("click", () => void build());
  finishButton.addEventListener("click", () => {
    finishButton.disabled = true;
    void session?.finish().finally(() => { finishButton.disabled = false; });
  });
  rebuildButton.addEventListener("click", () => void makeMesh());
  glbButton.addEventListener("click", () => glbBlob && download(glbBlob, "spatial-drop-scan.glb"));
  usdzButton.addEventListener("click", () => usdzBlob && download(usdzBlob, "spatial-drop-scan.usdz"));
  plyButton.addEventListener("click", async () => {
    if (!session?.trainer) return;
    download(await session.exportPlyBlob(), "spatial-drop-scan.ply");
  });
  studioButton.addEventListener("click", () => {
    if (!glbUrl) return;
    window.dispatchEvent(new CustomEvent("spatialdrop:openinstudio", { detail: { url: glbUrl, name: "Spatial Drop scan" } }));
    window.location.hash = "studio";
  });
  if (import.meta.env.DEV) {
    Object.assign(window, {
      spatialDropLoadSplat(saved: { data: Float32Array; n: number; dc: "sigmoid" | "sh"; cameras: ScanCamera[]; viewMetas?: ViewMeta[] }, files: File[] = []) {
        splat = { data: saved.data, n: saved.n, dc: saved.dc };
        cameras = saved.cameras;
        viewMetas = saved.viewMetas ?? [];
        scannedPhotos = files;
        return makeMesh();
      },
    });
  }

  // eighth-wall.ts enables it once a mesh exists
  arButton.disabled = true;
  window.addEventListener("spatialdrop:modechange", (event) => {
    if ((event as CustomEvent<string>).detail !== "scan") stopCamera();
  });
  window.addEventListener("pagehide", stopCamera);
  renderPhotos();
}
