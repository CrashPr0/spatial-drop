import { importModel } from "./model-import";
import "@google/model-viewer";
import QRCode from "qrcode";
import { setupBeamLab, stopBeamScanner } from "./beam";
import { setupEighthWall } from "./eighth-wall";
import { setupScanLab } from "./scan";
import "./style.css";

const dropZone = document.querySelector<HTMLDivElement>("#drop-zone")!;
const fileInput = document.querySelector<HTMLInputElement>("#file-input")!;
const modelUrl = document.querySelector<HTMLInputElement>("#model-url")!;
const modelTitle = document.querySelector<HTMLInputElement>("#model-title")!;
const resizable = document.querySelector<HTMLInputElement>("#resizable")!;
const viewer = document.querySelector<HTMLElement & { src: string }>("#viewer")!;
const empty = document.querySelector<HTMLElement>("#empty")!;
const status = document.querySelector<HTMLElement>("#status")!;
const error = document.querySelector<HTMLElement>("#error")!;
const shareButton = document.querySelector<HTMLButtonElement>("#share-button")!;
const sharePanel = document.querySelector<HTMLElement>("#share-panel")!;
let placement = "floor";
let localObjectUrl = "";

function setError(message = "") {
  error.textContent = message;
  error.hidden = !message;
}

function setModel(src: string) {
  viewer.setAttribute("src", src);
  empty.hidden = true;
  status.textContent = "MODEL LOADED";
}

let importId = 0;
async function acceptFile(file?: File) {
  if (!file) return;
  const id = ++importId;
  setError();
  try {
    const model = await importModel(file);
    if (id !== importId) return;
    if (localObjectUrl) URL.revokeObjectURL(localObjectUrl);
    localObjectUrl = URL.createObjectURL(model);
    setModel(localObjectUrl);
    document.querySelector("#drop-title")!.textContent = file.name;
    document.querySelector("#drop-detail")!.textContent = `${(model.size / 1024 / 1024).toFixed(1)} MB · Local preview${/\.obj$/i.test(file.name) ? " · OBJ converted; external materials not included" : ""}`;
    if (!modelTitle.value) modelTitle.value = file.name.replace(/\.(glb|obj|usdz)$/i, "").replace(/[-_]+/g, " ");
  } catch (reason) {
    if (id === importId) setError(reason instanceof Error ? reason.message : "Could not import this model.");
  }
}

dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") fileInput.click();
});
dropZone.addEventListener("dragover", (event) => { event.preventDefault(); dropZone.classList.add("dragging"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragging"));
dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragging");
  void acceptFile(event.dataTransfer?.files[0]);
});
fileInput.addEventListener("change", () => void acceptFile(fileInput.files?.[0]));

modelUrl.addEventListener("input", () => {
  const value = modelUrl.value.trim();
  shareButton.disabled = !value;
  if (value) setModel(value);
});

document.querySelectorAll<HTMLButtonElement>("[data-placement]").forEach((button) => {
  button.addEventListener("click", () => {
    placement = button.dataset.placement || "floor";
    document.querySelectorAll("[data-placement]").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    viewer.setAttribute("ar-placement", placement);
  });
});

resizable.addEventListener("change", () => viewer.setAttribute("ar-scale", resizable.checked ? "auto" : "fixed"));

function createShareUrl() {
  const url = new URL(window.location.href);
  url.search = "";
  url.searchParams.set("model", modelUrl.value.trim());
  url.searchParams.set("title", modelTitle.value.trim() || "Untitled model");
  url.searchParams.set("placement", placement);
  url.searchParams.set("resize", String(resizable.checked));
  return url.toString();
}

shareButton.addEventListener("click", async () => {
  try {
    const source = new URL(modelUrl.value.trim());
    if (source.protocol !== "https:") throw new Error("The public model URL must use HTTPS.");
    if (!source.pathname.toLowerCase().endsWith(".glb")) throw new Error("The public model URL must point to a .glb file.");
    setError();
    const shareUrl = createShareUrl();
    document.querySelector("#share-url")!.textContent = shareUrl;
    document.querySelector<HTMLAnchorElement>("#open-link")!.href = shareUrl;
    await QRCode.toCanvas(document.querySelector<HTMLCanvasElement>("#qr-canvas")!, shareUrl, { width: 360, margin: 2, color: { dark: "#0055A2", light: "#ffffff" }, errorCorrectionLevel: "H" });
    sharePanel.hidden = false;
    sharePanel.scrollIntoView({ behavior: "smooth", block: "center" });
  } catch (reason) {
    setError(reason instanceof Error ? reason.message : "Could not create that link.");
  }
});

const copyButton = document.querySelector<HTMLButtonElement>("#copy-button")!;
const shareUrlText = document.querySelector<HTMLElement>("#share-url")!;
const copyStatus = document.createElement("p");
copyStatus.setAttribute("role", "status");
copyStatus.setAttribute("aria-live", "polite");
copyStatus.setAttribute("aria-atomic", "true");
copyButton.closest(".actions")!.after(copyStatus);
let copyTimer: ReturnType<typeof setTimeout> | undefined;

copyButton.addEventListener("click", async () => {
  let copied = false;
  try {
    await navigator.clipboard.writeText(shareUrlText.textContent || "");
    copied = true;
  } catch {
    // Keep the displayed link selected if automatic copying is unavailable.
    try {
      const selection = window.getSelection();
      if (selection) {
        const range = document.createRange();
        range.selectNodeContents(shareUrlText);
        selection.removeAllRanges();
        selection.addRange(range);
        copied = document.execCommand("copy");
      }
    } catch {
      copied = false;
    }
  }
  clearTimeout(copyTimer);
  copyButton.textContent = copied ? "Copied!" : "Copy link";
  copyStatus.textContent = copied ? "Copied!" : "Please copy the link above manually.";
  if (copied) {
    copyTimer = setTimeout(() => {
      copyButton.textContent = "Copy link";
      copyStatus.textContent = "";
    }, 2000);
  }
});

window.addEventListener("pagehide", () => {
  clearTimeout(copyTimer);
  copyButton.textContent = "Copy link";
  copyStatus.textContent = "";
});

const initial = new URLSearchParams(window.location.search);
const initialModel = initial.get("model");
if (initialModel) {
  modelUrl.value = initialModel;
  modelTitle.value = initial.get("title") || "Untitled model";
  placement = initial.get("placement") === "wall" ? "wall" : "floor";
  resizable.checked = initial.get("resize") !== "false";
  viewer.setAttribute("ar-placement", placement);
  viewer.setAttribute("ar-scale", resizable.checked ? "auto" : "fixed");
  document.querySelectorAll("[data-placement]").forEach((item) => item.classList.toggle("active", (item as HTMLElement).dataset.placement === placement));
  setModel(initialModel);
  shareButton.disabled = false;
}

const modes = ["studio", "scan", "beam", "receive"] as const;
type Mode = typeof modes[number];

function setMode(mode: Mode) {
  modes.forEach((candidate) => {
    document.querySelector<HTMLElement>(`#${candidate}-mode`)!.hidden = candidate !== mode;
    document.querySelector<HTMLButtonElement>(`[data-mode="${candidate}"]`)!.classList.toggle("active", candidate === mode);
  });
  if (mode !== "receive") stopBeamScanner();
  window.dispatchEvent(new CustomEvent("spatialdrop:modechange", { detail: mode }));
}

function modeFromHash(): Mode {
  const value = window.location.hash.slice(1);
  return modes.includes(value as Mode) ? value as Mode : "studio";
}

document.querySelectorAll<HTMLButtonElement>("[data-mode]").forEach((button) => {
  button.addEventListener("click", () => {
    window.location.hash = button.dataset.mode || "studio";
  });
});
window.addEventListener("hashchange", () => setMode(modeFromHash()));

window.addEventListener("spatialdrop:openinstudio", (event) => {
  const { url, name } = (event as CustomEvent<{ url: string; name: string }>).detail;
  setError();
  setModel(url);
  document.querySelector("#drop-title")!.textContent = name;
  document.querySelector("#drop-detail")!.textContent = "Scanned on this device · Local preview";
  if (!modelTitle.value) modelTitle.value = name;
});

setupBeamLab();
setupScanLab();
setupEighthWall(stopBeamScanner);
setMode(modeFromHash());
