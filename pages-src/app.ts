import "@google/model-viewer";
import QRCode from "qrcode";
import "./style.css";

const MAX_BYTES = 15 * 1024 * 1024;
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

async function acceptFile(file?: File) {
  setError();
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".glb")) return setError("Please choose a binary glTF file ending in .glb.");
  if (file.size > MAX_BYTES) return setError("That model is over 15 MB. Optimize it, then try again.");
  const signature = new TextDecoder().decode((await file.slice(0, 4).arrayBuffer()));
  if (signature !== "glTF") return setError("This file is not a valid binary glTF model.");
  if (localObjectUrl) URL.revokeObjectURL(localObjectUrl);
  localObjectUrl = URL.createObjectURL(file);
  setModel(localObjectUrl);
  document.querySelector("#drop-title")!.textContent = file.name;
  document.querySelector("#drop-detail")!.textContent = `${(file.size / 1024 / 1024).toFixed(1)} MB · Local preview`;
  if (!modelTitle.value) modelTitle.value = file.name.replace(/\.glb$/i, "").replace(/[-_]+/g, " ");
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

document.querySelector("#copy-button")!.addEventListener("click", async () => navigator.clipboard.writeText(createShareUrl()));

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
