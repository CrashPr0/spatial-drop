"use client";

import Link from "next/link";
import { ChangeEvent, DragEvent, useEffect, useRef, useState } from "react";
import { ModelViewer } from "./ModelViewer";

const MAX_FILE_BYTES = 80 * 1024 * 1024;
const EIGHTH_WALL_ENGINE = "https://cdn.jsdelivr.net/npm/@8thwall/engine-binary@1/dist/xr.js";
const EIGHT_FRAME = "https://cdn.8thwall.com/web/aframe/8frame-1.5.0.min.js";
const XR_EXTRAS = "https://cdn.jsdelivr.net/npm/@8thwall/xrextras@1/dist/xrextras.js";

type ProcessingState = "idle" | "reading" | "converting" | "ready" | "error";

declare global {
  interface Window {
    AFRAME?: {
      registerComponent: (name: string, definition: Record<string, unknown>) => void;
      components?: Record<string, unknown>;
      THREE: {
        Box3: new () => {
          setFromObject: (object: unknown) => { min: { y: number } };
        };
      };
    };
    XR8?: {
      XrConfig: { device: () => { MOBILE: unknown } };
      XrDevice: { isDeviceBrowserCompatible: (options: { allowedDevices: unknown }) => boolean };
      XrController: { recenter: () => void };
      stop?: () => void;
      pause?: () => void;
    };
  }
}

function formatBytes(bytes: number) {
  return bytes < 1024 * 1024
    ? `${Math.ceil(bytes / 1024)} KB`
    : `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function loadScript(id: string, src: string, attributes: Record<string, string> = {}) {
  return new Promise<void>((resolve, reject) => {
    const existing = document.getElementById(id) as HTMLScriptElement | null;
    if (existing?.dataset.loaded === "true") return resolve();
    if (existing) {
      existing.addEventListener("load", () => resolve(), { once: true });
      existing.addEventListener("error", () => reject(new Error(`Could not load ${src}`)), { once: true });
      return;
    }
    const script = document.createElement("script");
    script.id = id;
    script.src = src;
    script.crossOrigin = "anonymous";
    Object.entries(attributes).forEach(([key, value]) => script.setAttribute(key, value));
    script.addEventListener("load", () => {
      script.dataset.loaded = "true";
      resolve();
    }, { once: true });
    script.addEventListener("error", () => reject(new Error(`Could not load ${src}`)), { once: true });
    document.head.appendChild(script);
  });
}

function waitForEighthWall(timeoutMs = 30000) {
  return new Promise<void>((resolve, reject) => {
    let timeout = 0;
    let interval = 0;

    const cleanup = () => {
      window.clearTimeout(timeout);
      window.clearInterval(interval);
      window.removeEventListener("xrloaded", checkReady);
    };
    const checkReady = () => {
      if (!window.XR8 || !window.AFRAME) return;
      cleanup();
      resolve();
    };

    window.addEventListener("xrloaded", checkReady);
    interval = window.setInterval(checkReady, 50);
    timeout = window.setTimeout(() => {
      cleanup();
      reject(new Error("The 8th Wall runtime did not initialize."));
    }, timeoutMs);
    checkReady();
  });
}

export function AppleScanPreview() {
  const [sourceFile, setSourceFile] = useState<File | null>(null);
  const [glbUrl, setGlbUrl] = useState("");
  const [usdzUrl, setUsdzUrl] = useState("");
  const [processing, setProcessing] = useState<ProcessingState>("idle");
  const [message, setMessage] = useState("Choose a USDZ from Apple Object Capture to begin.");
  const [dragging, setDragging] = useState(false);
  const [arActive, setArActive] = useState(false);
  const [arStatus, setArStatus] = useState("Preparing 8th Wall…");
  const [arError, setArError] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const arRootRef = useRef<HTMLDivElement>(null);
  const urlsRef = useRef<string[]>([]);

  useEffect(() => {
    return () => {
      urlsRef.current.forEach((url) => URL.revokeObjectURL(url));
      if (window.XR8?.stop) window.XR8.stop();
    };
  }, []);

  useEffect(() => {
    if (!arActive || !glbUrl || !arRootRef.current || !window.AFRAME) return;
    const root = arRootRef.current;
    const onReady = () => setArStatus("Tracking ready · model placed ahead");
    const onError = () => setArError("8th Wall could not start the camera. Check camera and motion permissions.");
    window.addEventListener("realityready", onReady, { once: true });
    window.addEventListener("realityerror", onError, { once: true });

    const componentName = "spatial-drop-ground-model";
    if (!window.AFRAME.components?.[componentName]) {
      window.AFRAME.registerComponent(componentName, {
        init(this: { el: { addEventListener: (name: string, callback: () => void, options: { once: boolean }) => void; getObject3D: (name: string) => { position: { y: number } } | null } }) {
          this.el.addEventListener("model-loaded", () => {
            const model = this.el.getObject3D("mesh");
            if (!model || !window.AFRAME) return;
            const box = new window.AFRAME.THREE.Box3().setFromObject(model);
            model.position.y -= box.min.y;
          }, { once: true });
        },
      });
    }

    const scene = document.createElement("a-scene");
    scene.setAttribute("xrweb", "");
    scene.setAttribute("xrconfig", "cameraDirection: back");
    scene.setAttribute("xrextras-loading", "");
    scene.setAttribute("xrextras-runtime-error", "");
    scene.setAttribute("renderer", "colorManagement: true; physicallyCorrectLights: true; antialias: true");
    scene.setAttribute("embedded", "");
    scene.className = "eighth-wall-scene";

    const light = document.createElement("a-entity");
    light.setAttribute("light", "type: directional; intensity: 1.2; castShadow: true");
    light.setAttribute("position", "2 5 3");
    scene.appendChild(light);

    const ambient = document.createElement("a-entity");
    ambient.setAttribute("light", "type: ambient; intensity: 0.7");
    scene.appendChild(ambient);

    const model = document.createElement("a-entity");
    model.setAttribute("gltf-model", `url(${glbUrl})`);
    model.setAttribute("position", "0 0 -2.5");
    model.setAttribute("shadow", "cast: true; receive: false");
    model.setAttribute(componentName, "");
    scene.appendChild(model);

    const ground = document.createElement("a-plane");
    ground.setAttribute("position", "0 0 -2.5");
    ground.setAttribute("rotation", "-90 0 0");
    ground.setAttribute("width", "200");
    ground.setAttribute("height", "200");
    ground.setAttribute("material", "shader: shadow; opacity: 0.35");
    ground.setAttribute("shadow", "receive: true");
    scene.appendChild(ground);

    const camera = document.createElement("a-camera");
    camera.id = "camera";
    camera.setAttribute("position", "0 1.6 0");
    scene.appendChild(camera);
    root.appendChild(scene);

    return () => {
      window.removeEventListener("realityready", onReady);
      window.removeEventListener("realityerror", onError);
      scene.remove();
    };
  }, [arActive, glbUrl]);

  function rememberUrl(url: string) {
    urlsRef.current.push(url);
    return url;
  }

  function clearPreviousModel() {
    urlsRef.current.forEach((url) => URL.revokeObjectURL(url));
    urlsRef.current = [];
    setGlbUrl("");
    setUsdzUrl("");
    setArError("");
    setArActive(false);
  }

  async function processFile(file?: File) {
    if (!file) return;
    clearPreviousModel();
    setSourceFile(file);
    const extension = file.name.split(".").pop()?.toLowerCase();
    if (extension !== "usdz" && extension !== "glb") {
      setProcessing("error");
      setMessage("Choose an Apple Object Capture .usdz file or a browser-ready .glb.");
      return;
    }
    if (file.size > MAX_FILE_BYTES) {
      setProcessing("error");
      setMessage("This prototype accepts files up to 80 MB. Export a Reduced or Medium Object Capture model.");
      return;
    }

    try {
      setProcessing("reading");
      setMessage(`Reading ${file.name}…`);
      const buffer = await file.arrayBuffer();

      if (extension === "glb") {
        const signature = new TextDecoder().decode(buffer.slice(0, 4));
        if (signature !== "glTF") throw new Error("That file is not a valid binary glTF model.");
        setGlbUrl(rememberUrl(URL.createObjectURL(new Blob([buffer], { type: "model/gltf-binary" }))));
      } else {
        setUsdzUrl(rememberUrl(URL.createObjectURL(file)));
        setProcessing("converting");
        setMessage("Converting USDZ materials and geometry for the browser…");
        const [{ USDLoader }, { GLTFExporter }] = await Promise.all([
          import("three/addons/loaders/USDLoader.js"),
          import("three/addons/exporters/GLTFExporter.js"),
        ]);
        const group = new USDLoader().parse(buffer);
        const exported = await new GLTFExporter().parseAsync(group, {
          binary: true,
          onlyVisible: true,
          maxTextureSize: 2048,
        });
        if (!(exported instanceof ArrayBuffer)) throw new Error("USDZ conversion did not produce a GLB.");
        setGlbUrl(rememberUrl(URL.createObjectURL(new Blob([exported], { type: "model/gltf-binary" }))));
      }

      setProcessing("ready");
      setMessage(extension === "usdz"
        ? "USDZ converted locally · ready for browser preview and 8th Wall AR."
        : "GLB validated · ready for browser preview and 8th Wall AR.");
    } catch (reason) {
      setProcessing("error");
      setMessage(reason instanceof Error ? reason.message : "The model could not be prepared.");
    }
  }

  async function startEighthWall() {
    if (!glbUrl) return;
    setArError("");
    setArStatus("Loading 8th Wall world tracking…");
    try {
      await loadScript("spatial-drop-eight-frame", EIGHT_FRAME);
      await loadScript("spatial-drop-xrextras", XR_EXTRAS);
      await loadScript("spatial-drop-xr8", EIGHTH_WALL_ENGINE, { "data-preload-chunks": "slam" });
      await waitForEighthWall();
      const mobile = window.XR8.XrConfig.device().MOBILE;
      if (!window.XR8.XrDevice.isDeviceBrowserCompatible({ allowedDevices: mobile })) {
        throw new Error("8th Wall world tracking needs a compatible iPhone or Android browser. The 3D preview still works here.");
      }
      setArActive(true);
      setArStatus("Allow camera access, then move your phone slowly");
    } catch (reason) {
      setArError(reason instanceof Error ? reason.message : "8th Wall could not start.");
    }
  }

  function exitAR() {
    if (window.XR8?.stop) window.XR8.stop();
    else if (window.XR8?.pause) window.XR8.pause();
    setArActive(false);
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    void processFile(event.dataTransfer.files[0]);
  }

  return (
    <main className="scan-preview-shell">
      <header className="site-header scan-header">
        <Link className="brand" href="/" aria-label="Spatial Drop preview home">
          <span className="brand-glyph brand-glyph-sjsu" aria-hidden="true">SJSU</span><span className="brand-copy"><strong>SPATIAL DROP</strong><small>XR PROTOTYPING LAB</small></span>
        </Link>
        <div className="scan-nav"><span className="prototype-pill">SPARTAN INNOVATION · 8TH WALL</span><Link href="/publish">Publisher prototype ↗</Link></div>
      </header>

      <section className="scan-hero">
        <div><p className="eyebrow">SAN JOSÉ STATE · SPATIAL COMPUTING</p><h1>Scan it.<br />Place it.</h1></div>
        <div className="scan-hero-copy"><p>Turn an Apple USDZ scan into an interactive browser preview, then place it in the room with 8th Wall world tracking.</p><span>PRIVATE BY DESIGN · NOTHING UPLOADS</span></div>
      </section>

      <section className="scan-workspace">
        <div className="scan-controls">
          <div className="step-heading"><span>01</span><div><h2>Choose an Apple scan</h2><p>USDZ preferred · GLB accepted · 80 MB maximum</p></div></div>
          <div
            className={`scan-drop-zone ${dragging ? "is-dragging" : ""} ${sourceFile ? "has-file" : ""}`}
            role="button"
            tabIndex={0}
            onClick={() => inputRef.current?.click()}
            onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") inputRef.current?.click(); }}
            onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
          >
            <input ref={inputRef} className="visually-hidden" type="file" accept=".usdz,.glb,model/vnd.usdz+zip,model/gltf-binary" onChange={(event: ChangeEvent<HTMLInputElement>) => void processFile(event.target.files?.[0])} />
            <span className="scan-file-mark" aria-hidden="true">USDZ</span>
            <div><strong>{sourceFile?.name || "Drop an Object Capture model"}</strong><span>{sourceFile ? `${formatBytes(sourceFile.size)} · ${processing}` : "or click to choose .usdz / .glb"}</span></div>
          </div>

          <div className={`conversion-status status-${processing}`}>
            <span className="status-dot" aria-hidden="true" /><div><strong>{processing === "ready" ? "Model ready" : processing === "error" ? "Could not prepare model" : processing === "idle" ? "Waiting for model" : "Processing on this device"}</strong><p>{message}</p></div>
          </div>

          <div className="scan-facts"><div><span>FORMAT</span><strong>{sourceFile?.name.toLowerCase().endsWith(".usdz") ? "USDZ → GLB" : sourceFile ? "GLB" : "—"}</strong></div><div><span>RUNTIME</span><strong>8th Wall SLAM</strong></div><div><span>PRIVACY</span><strong>Local only</strong></div></div>

          <button className="publish-button scan-ar-button" type="button" disabled={!glbUrl || processing !== "ready"} onClick={startEighthWall}><span>Start 8th Wall AR</span><span aria-hidden="true">→</span></button>
          {arError && <p className="error-message" role="alert">{arError}</p>}
          {usdzUrl && <a className="quick-look-link" rel="ar" href={usdzUrl}>Open original USDZ in Apple Quick Look ↗</a>}
          <p className="scan-note">8th Wall world tracking runs on compatible mobile browsers. Desktop uses the interactive 3D preview.</p>
        </div>

        <div className="scan-stage-panel">
          <div className="preview-topline"><span>BROWSER PREVIEW</span><span>{glbUrl ? "CONVERTED MODEL" : "NO MODEL"}</span></div>
          <div className="scan-model-stage">
            {glbUrl ? <ModelViewer src={glbUrl} iosSrc={usdzUrl || undefined} alt={`Preview of ${sourceFile?.name || "Apple scan"}`} className="scan-model-viewer" /> : <div className="scan-empty"><div className="scan-rings"><i /><i /><i /></div><p>YOUR APPLE SCAN<br />WILL APPEAR HERE</p></div>}
          </div>
          <div className="preview-footer"><span>Drag to orbit</span><span>Scroll to zoom</span><span>1 unit = 1 meter</span></div>
        </div>
      </section>

      <section className="scan-pipeline"><p className="eyebrow">FROM SCAN TO SPACE</p><div><article><span>01</span><h3>Read USDZ</h3><p>Parse Apple’s packaged geometry, textures, and physical scale.</p></article><article><span>02</span><h3>Convert locally</h3><p>Create a temporary browser-ready GLB without uploading the scan.</p></article><article><span>03</span><h3>Track the room</h3><p>Use 8th Wall SLAM to place the converted model on the ground plane.</p></article></div></section>

      <footer><span>SPATIAL DROP · SJSU-INSPIRED XR PROTOTYPE</span><span>This product includes the XR Engine by Niantic Spatial, Inc. · <a href="https://github.com/8thwall/engine/blob/main/LICENSE">License</a></span></footer>

      {arActive && <div className="eighth-wall-overlay"><div ref={arRootRef} className="eighth-wall-root" /><div className="ar-hud"><div><span className="live-dot" />{arStatus}</div><div><button type="button" onClick={() => window.XR8?.XrController.recenter()}>Recenter</button><button type="button" onClick={exitAR}>Exit AR</button></div></div></div>}
    </main>
  );
}
