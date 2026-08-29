"use client";

import QRCode from "qrcode";
import Image from "next/image";
import Link from "next/link";
import { ChangeEvent, DragEvent, useEffect, useMemo, useRef, useState } from "react";
import { ModelViewer } from "./ModelViewer";

type Placement = "floor" | "wall";

type PublishedModel = {
  id: string;
  title: string;
  viewUrl: string;
  modelUrl: string;
};

const MAX_BYTES = 15 * 1024 * 1024;

function formatBytes(bytes: number) {
  return bytes < 1024 * 1024
    ? `${Math.ceil(bytes / 1024)} KB`
    : `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export function UploadStudio() {
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState("");
  const [placement, setPlacement] = useState<Placement>("floor");
  const [resizable, setResizable] = useState(true);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");
  const [publishing, setPublishing] = useState(false);
  const [published, setPublished] = useState<PublishedModel | null>(null);
  const [qrDataUrl, setQrDataUrl] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const previewUrl = useMemo(() => (file ? URL.createObjectURL(file) : ""), [file]);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  useEffect(() => {
    if (!published) return;
    QRCode.toDataURL(published.viewUrl, {
      width: 420,
      margin: 2,
      color: { dark: "#171713", light: "#fffef8" },
      errorCorrectionLevel: "H",
    }).then(setQrDataUrl);
  }, [published]);

  function acceptFile(candidate?: File) {
    setError("");
    setPublished(null);
    if (!candidate) return;
    if (!candidate.name.toLowerCase().endsWith(".glb")) {
      setError("Please choose a binary glTF file ending in .glb.");
      return;
    }
    if (candidate.size > MAX_BYTES) {
      setError("That model is over 15 MB. Optimize it, then try again.");
      return;
    }
    setFile(candidate);
    if (!title) setTitle(candidate.name.replace(/\.glb$/i, "").replace(/[-_]+/g, " "));
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    acceptFile(event.dataTransfer.files[0]);
  }

  async function publish() {
    if (!file) return;
    setPublishing(true);
    setError("");
    try {
      const form = new FormData();
      form.append("model", file);
      form.append("title", title.trim() || "Untitled model");
      form.append("placement", placement);
      form.append("resizable", String(resizable));
      const response = await fetch("/api/models", { method: "POST", body: form });
      const payload = (await response.json()) as { id?: string; error?: string };
      if (!response.ok || !payload.id) throw new Error(payload.error || "Publishing failed.");

      const viewUrl = `${window.location.origin}/view/${payload.id}`;
      setPublished({
        id: payload.id,
        title: title.trim() || "Untitled model",
        viewUrl,
        modelUrl: `${window.location.origin}/api/models/${payload.id}/file`,
      });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Publishing failed. Please try again.");
    } finally {
      setPublishing(false);
    }
  }

  async function copyLink() {
    if (!published) return;
    await navigator.clipboard.writeText(published.viewUrl);
  }

  function downloadQr() {
    if (!qrDataUrl || !published) return;
    const link = document.createElement("a");
    link.href = qrDataUrl;
    link.download = `${published.title.toLowerCase().replace(/[^a-z0-9]+/g, "-")}-qr.png`;
    link.click();
  }

  return (
    <main className="studio-shell">
      <header className="site-header">
        <Link className="brand" href="/" aria-label="Spatial Drop home">
          <span className="brand-glyph" aria-hidden="true">S/</span>
          <span>SPATIAL DROP</span>
        </Link>
        <span className="prototype-pill">WORKING PROTOTYPE</span>
      </header>

      <section className="intro">
        <p className="eyebrow">3D → URL → REAL SPACE</p>
        <h1>Put your model<br />in the room.</h1>
        <p className="lede">
          Upload one GLB. We turn it into a shareable AR page and a QR code—no app download required.
        </p>
      </section>

      <section className="workspace" aria-label="AR publishing studio">
        <div className="control-panel">
          <div className="step-heading">
            <span>01</span>
            <div><h2>Choose a model</h2><p>Binary glTF · 15 MB maximum</p></div>
          </div>

          <div
            className={`drop-zone ${dragging ? "is-dragging" : ""} ${file ? "has-file" : ""}`}
            onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => inputRef.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") inputRef.current?.click(); }}
          >
            <input
              ref={inputRef}
              className="visually-hidden"
              type="file"
              accept=".glb,model/gltf-binary"
              onChange={(event: ChangeEvent<HTMLInputElement>) => acceptFile(event.target.files?.[0])}
            />
            <span className="upload-mark" aria-hidden="true">↗</span>
            {file ? (
              <div><strong>{file.name}</strong><span>{formatBytes(file.size)} · Ready to preview</span></div>
            ) : (
              <div><strong>Drop a GLB here</strong><span>or click to choose a file</span></div>
            )}
          </div>

          <div className="form-stack">
            <label>
              <span>Display name</span>
              <input value={title} onChange={(event) => setTitle(event.target.value)} placeholder="Eames lounge chair" maxLength={80} />
            </label>

            <fieldset>
              <legend>Placement surface</legend>
              <div className="segmented-control">
                <button type="button" className={placement === "floor" ? "active" : ""} onClick={() => setPlacement("floor")}>Floor</button>
                <button type="button" className={placement === "wall" ? "active" : ""} onClick={() => setPlacement("wall")}>Wall</button>
              </div>
            </fieldset>

            <div className="toggle-row">
              <span id="resize-label"><strong>Allow pinch to resize</strong><small>Otherwise the GLB’s meter scale stays fixed.</small></span>
              <input aria-labelledby="resize-label" type="checkbox" checked={resizable} onChange={(event) => setResizable(event.target.checked)} />
            </div>
          </div>

          {error && <p className="error-message" role="alert">{error}</p>}

          <button className="publish-button" type="button" disabled={!file || publishing} onClick={publish}>
            <span>{publishing ? "Publishing…" : "Publish AR page"}</span><span aria-hidden="true">→</span>
          </button>
          <p className="scale-note">For accurate size, author your GLB at 1 unit = 1 meter.</p>
        </div>

        <div className="preview-panel">
          <div className="preview-topline"><span>LIVE PREVIEW</span><span>{file ? "MODEL LOADED" : "WAITING FOR MODEL"}</span></div>
          <div className="viewer-stage">
            {previewUrl ? (
              <ModelViewer src={previewUrl} alt={`Preview of ${title || file?.name}`} placement={placement} resizable={resizable} className="model-viewer" />
            ) : (
              <div className="empty-stage" aria-hidden="true">
                <div className="axis axis-x"><span>X</span></div>
                <div className="axis axis-y"><span>Y</span></div>
                <div className="axis axis-z"><span>Z</span></div>
                <div className="wire-cube"><i /><i /><i /><i /></div>
                <p>Your model will appear here</p>
              </div>
            )}
          </div>
          <div className="preview-footer"><span>Drag to orbit</span><span>Scroll to zoom</span><span>AR available after publish</span></div>
        </div>
      </section>

      {published && (
        <section className="success-panel" aria-live="polite">
          <div className="success-copy">
            <p className="eyebrow">PUBLISHED</p>
            <h2>Your model is ready<br />for the real world.</h2>
            <p>Scan with an iPhone or Android camera, then tap “View in your space.”</p>
            <div className="link-box"><span>{published.viewUrl}</span><button type="button" onClick={copyLink}>Copy link</button></div>
            <div className="success-actions">
              <a href={published.viewUrl}>Open AR page ↗</a>
              <button type="button" onClick={downloadQr}>Download QR</button>
            </div>
          </div>
          <div className="qr-card">
            {qrDataUrl && <Image unoptimized width={420} height={420} src={qrDataUrl} alt={`QR code for ${published.title}`} />}
            <div><strong>SCAN TO PLACE</strong><span>{published.title}</span></div>
          </div>
        </section>
      )}

      <footer><span>SPATIAL DROP / PROTOTYPE 01</span><span>GLB · QUICK LOOK · SCENE VIEWER · WEBXR</span></footer>
    </main>
  );
}
