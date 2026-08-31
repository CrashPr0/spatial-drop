(function () {
  "use strict";

  const embedded = new URLSearchParams(window.location.search).get("embed") === "1";
  const parentOrigin = window.location.origin;
  let role = "";

  function post(message, transfer) {
    if (!embedded || window.parent === window) return;
    window.parent.postMessage(message, parentOrigin, transfer || []);
  }

  function applyEmbeddedStyle(nextRole) {
    if (!embedded) return;
    document.documentElement.classList.add("spatial-drop-embed", `spatial-drop-${nextRole}`);
    const style = document.createElement("style");
    style.textContent = `
      .spatial-drop-embed body { background: #061a30; overflow: hidden; }
      .spatial-drop-embed #nav-container,
      .spatial-drop-embed #debug_menu,
      .spatial-drop-embed #status,
      .spatial-drop-embed .drop-message { display: none !important; }
      .spatial-drop-sender #dragdrop { width: 100vw; height: 100vh; outline: 0; box-shadow: none; background: #fff; }
      .spatial-drop-sender #canvas { margin: auto; }
      .spatial-drop-receiver #progress_bars { display: none; }
    `;
    document.head.appendChild(style);
  }

  function patchReceiver() {
    if (!window.Recv || !window.Zstd) return;
    const originalProgress = Recv.render_progress.bind(Recv);
    Recv.render_progress = function (report) {
      originalProgress(report);
      post({ type: "spatialdrop:cimbar-progress", report: Array.from(report || []) });
    };

    const originalDownload = Zstd.download_blob.bind(Zstd);
    Zstd.download_blob = function (name, blob) {
      if (!embedded) {
        originalDownload(name, blob);
        return;
      }
      blob.arrayBuffer().then(function (buffer) {
        post({ type: "spatialdrop:cimbar-model", name: name, buffer: buffer }, [buffer]);
      }).catch(function (error) {
        post({ type: "spatialdrop:cimbar-error", message: error instanceof Error ? error.message : String(error) });
      });
    };
  }

  window.SpatialDropCimbar = {
    ready: function (nextRole) {
      role = nextRole;
      applyEmbeddedStyle(role);
      if (role === "receiver") patchReceiver();
      post({ type: "spatialdrop:cimbar-ready", role: role });
    }
  };

  window.addEventListener("message", function (event) {
    if (event.origin !== parentOrigin || event.source !== window.parent || !event.data) return;
    if (role === "sender" && event.data.type === "spatialdrop:cimbar-send" && event.data.file instanceof File) {
      Main.setMode(event.data.mode || "Bm");
      Main.setFPS(event.data.fps || 10);
      Main.importFile(event.data.file);
    } else if (role === "sender" && event.data.type === "spatialdrop:cimbar-pause") {
      Main.togglePause(Boolean(event.data.paused));
    } else if (role === "receiver" && event.data.type === "spatialdrop:cimbar-stop") {
      const video = document.querySelector("video");
      if (video) {
        video.pause();
        if (video.srcObject) video.srcObject.getTracks().forEach(function (track) { track.stop(); });
        video.srcObject = null;
      }
    }
  });
})();
