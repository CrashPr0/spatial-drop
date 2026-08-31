const EIGHTH_WALL_ENGINE = "https://cdn.jsdelivr.net/npm/@8thwall/engine-binary@1/dist/xr.js";
const EIGHT_FRAME = "https://cdn.8thwall.com/web/aframe/8frame-1.5.0.min.js";
const XR_EXTRAS = "https://cdn.jsdelivr.net/npm/@8thwall/xrextras@1/dist/xrextras.js";

type AFrameRuntime = {
  components?: Record<string, unknown>;
  registerComponent: (name: string, definition: Record<string, unknown>) => void;
  THREE: {
    Box3: new () => { setFromObject: (object: unknown) => { min: { y: number } } };
  };
};

type XR8Runtime = {
  XrConfig: { device: () => { MOBILE: unknown } };
  XrDevice: { isDeviceBrowserCompatible: (options: { allowedDevices: unknown }) => boolean };
  XrController: { recenter: () => void };
  stop?: () => void;
  pause?: () => void;
};

type SpatialWindow = Window & typeof globalThis & { AFRAME?: AFrameRuntime; XR8?: XR8Runtime };

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
    script.addEventListener("load", () => { script.dataset.loaded = "true"; resolve(); }, { once: true });
    script.addEventListener("error", () => reject(new Error(`Could not load ${src}`)), { once: true });
    document.head.appendChild(script);
  });
}

function waitForRuntime(runtimeWindow: SpatialWindow, timeoutMs = 30000) {
  return new Promise<void>((resolve, reject) => {
    let timeout = 0;
    let interval = 0;
    const cleanup = () => {
      window.clearTimeout(timeout);
      window.clearInterval(interval);
      window.removeEventListener("xrloaded", ready);
    };
    const ready = () => {
      if (!runtimeWindow.XR8 || !runtimeWindow.AFRAME) return;
      cleanup();
      resolve();
    };
    window.addEventListener("xrloaded", ready);
    interval = window.setInterval(ready, 50);
    timeout = window.setTimeout(() => {
      cleanup();
      reject(new Error("The 8th Wall runtime did not initialize."));
    }, timeoutMs);
    ready();
  });
}

function setError(element: HTMLElement, message = "") {
  element.textContent = message;
  element.hidden = !message;
}

export function setupEighthWall(stopScanner: () => void) {
  const runtimeWindow = window as SpatialWindow;
  const launchButton = document.querySelector<HTMLButtonElement>("#beam-ar-button")!;
  const error = document.querySelector<HTMLElement>("#beam-ar-error")!;
  const overlay = document.querySelector<HTMLElement>("#beam-ar-overlay")!;
  const root = document.querySelector<HTMLElement>("#beam-ar-root")!;
  const status = document.querySelector<HTMLElement>("#beam-ar-status")!;
  const recenterButton = document.querySelector<HTMLButtonElement>("#beam-ar-recenter")!;
  const exitButton = document.querySelector<HTMLButtonElement>("#beam-ar-exit")!;
  let modelUrl = "";

  window.addEventListener("spatialdrop:modelreceived", (event) => {
    modelUrl = (event as CustomEvent<string>).detail;
    launchButton.disabled = !modelUrl;
  });

  function exitAR() {
    if (runtimeWindow.XR8?.stop) runtimeWindow.XR8.stop();
    else if (runtimeWindow.XR8?.pause) runtimeWindow.XR8.pause();
    root.replaceChildren();
    overlay.hidden = true;
  }

  function mountScene() {
    const aframe = runtimeWindow.AFRAME;
    if (!aframe || !modelUrl) return;
    const componentName = "spatial-drop-ground-model";
    if (!aframe.components?.[componentName]) {
      aframe.registerComponent(componentName, {
        init(this: { el: { addEventListener: (name: string, callback: () => void, options: { once: boolean }) => void; getObject3D: (name: string) => { position: { y: number } } | null } }) {
          this.el.addEventListener("model-loaded", () => {
            const model = this.el.getObject3D("mesh");
            if (!model || !runtimeWindow.AFRAME) return;
            const box = new runtimeWindow.AFRAME.THREE.Box3().setFromObject(model);
            model.position.y -= box.min.y;
          }, { once: true });
        },
      });
    }

    root.replaceChildren();
    const scene = document.createElement("a-scene");
    scene.setAttribute("xrweb", "scale: absolute");
    scene.setAttribute("xrconfig", "cameraDirection: back");
    scene.setAttribute("xrextras-loading", "");
    scene.setAttribute("xrextras-runtime-error", "");
    scene.setAttribute("renderer", "colorManagement: true; physicallyCorrectLights: true; antialias: true");
    scene.setAttribute("embedded", "");

    const directionalLight = document.createElement("a-entity");
    directionalLight.setAttribute("light", "type: directional; intensity: 1.2; castShadow: true");
    directionalLight.setAttribute("position", "2 5 3");
    scene.appendChild(directionalLight);

    const ambientLight = document.createElement("a-entity");
    ambientLight.setAttribute("light", "type: ambient; intensity: 0.7");
    scene.appendChild(ambientLight);

    const model = document.createElement("a-entity");
    model.setAttribute("gltf-model", `url(${modelUrl})`);
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
    camera.setAttribute("position", "0 1.6 0");
    scene.appendChild(camera);
    root.appendChild(scene);
  }

  async function startAR() {
    if (!modelUrl) return;
    stopScanner();
    setError(error);
    status.innerHTML = "<i></i>Loading 8th Wall world tracking…";
    try {
      await loadScript("spatial-drop-eight-frame", EIGHT_FRAME);
      await loadScript("spatial-drop-xrextras", XR_EXTRAS);
      await loadScript("spatial-drop-xr8", EIGHTH_WALL_ENGINE, { "data-preload-chunks": "slam" });
      await waitForRuntime(runtimeWindow);
      if (!runtimeWindow.XR8) throw new Error("The 8th Wall runtime did not initialize.");
      const mobile = runtimeWindow.XR8.XrConfig.device().MOBILE;
      if (!runtimeWindow.XR8.XrDevice.isDeviceBrowserCompatible({ allowedDevices: mobile })) {
        throw new Error("8th Wall world tracking requires a compatible iPhone or Android browser. The reconstructed 3D preview still works here.");
      }
      overlay.hidden = false;
      status.innerHTML = "<i></i>Allow camera access, then move slowly";
      window.addEventListener("realityready", () => { status.innerHTML = "<i></i>Tracking ready · model placed ahead"; }, { once: true });
      window.addEventListener("realityerror", () => setError(error, "8th Wall could not start the camera. Check camera and motion permissions."), { once: true });
      mountScene();
    } catch (reason) {
      overlay.hidden = true;
      setError(error, reason instanceof Error ? reason.message : "8th Wall could not start.");
    }
  }

  launchButton.addEventListener("click", () => void startAR());
  recenterButton.addEventListener("click", () => runtimeWindow.XR8?.XrController.recenter());
  exitButton.addEventListener("click", exitAR);
  window.addEventListener("pagehide", exitAR);
}
