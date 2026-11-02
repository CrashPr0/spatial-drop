// Synthetic object-orbit capture for exercising Scan mode end to end.
// With the Pages dev server running (npx vite --config pages.vite.config.ts), in the page console:
//   await import('/@fs' + '<absolute repo path>/tests/browser/scan-orbit.js');
//   await makeOrbit();            // renders 44 views of a known scene and starts a build
//   scanStatus();                 // progress
//   await saveScan();             // keep the trained splat in IndexedDB
//   await makeOrbit({ build: false });               // regenerate the same photos (deterministic)
//   await spatialDropLoadSplat(await loadSaved(), __orbitFiles);   // re-mesh and re-texture without retraining
// Ground truth: a 0.18 x 0.26 x 0.14 m box (turned 0.5 rad) with a 0.08 m sphere on top,
// cameras on rings of radius 0.62 m / 0.55 m at heights 0.22 m / 0.48 m looking at y = 0.17 m.
window.makeOrbit = async function makeOrbit({ build = true } = {}) {
  const THREE = await import(new URL('../../node_modules/three/build/three.module.js', import.meta.url).href);
  let seed = 7; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  function tex(base, n, size = 1024) {
    const c = document.createElement('canvas'); c.width = c.height = size; const g = c.getContext('2d');
    g.fillStyle = base; g.fillRect(0, 0, size, size);
    // soft blobs plus a few crisp marks: textured like real surfaces, not confetti
    for (let i = 0; i < n; i++) {
      const x = rnd()*size, y = rnd()*size, r = size * (0.03 + rnd()*0.12);
      const grad = g.createRadialGradient(x, y, 0, x, y, r);
      grad.addColorStop(0, `hsla(${rnd()*360},${40+rnd()*40}%,${30+rnd()*45}%,0.9)`); grad.addColorStop(1, 'hsla(0,0%,50%,0)');
      g.fillStyle = grad; g.fillRect(x - r, y - r, 2*r, 2*r);
    }
    for (let i = 0; i < n / 3; i++) { g.fillStyle = `hsl(${rnd()*360},60%,${20+rnd()*60}%)`; g.fillRect(rnd()*size, rnd()*size, 6 + rnd()*size/30, 6 + rnd()*size/30); }
    const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
  }
  const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setSize(960, 720, false);
  const scene = new THREE.Scene();
  scene.add(new THREE.AmbientLight(0xffffff, 1.2));
  const sun = new THREE.DirectionalLight(0xffffff, 1.6); sun.position.set(1, 2, 1.5); scene.add(sun);
  // BoxGeometry repeats one texture on all six faces: give each face its own, or
  // opposite views become mirror images and the solver folds the orbit
  const faces = (base, n, size, side) => Array.from({ length: 6 }, () => new THREE.MeshStandardMaterial({ map: tex(base, n, size), side }));
  const room = new THREE.Mesh(new THREE.BoxGeometry(6, 4, 6), faces('#d8d4cc', 160, 1024, THREE.BackSide));
  room.position.y = 1.5; scene.add(room);
  const table = new THREE.Mesh(new THREE.PlaneGeometry(1.4, 1.4), new THREE.MeshStandardMaterial({ map: tex('#9a7b5a', 220) }));
  table.rotation.x = -Math.PI / 2; scene.add(table);
  const body = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.26, 0.14), faces('#c33', 90, 512, THREE.FrontSide));
  body.position.y = 0.13; scene.add(body);
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.08, 48, 32), new THREE.MeshStandardMaterial({ map: tex('#36c', 70, 512) }));
  head.position.y = 0.26 + 0.07; scene.add(head);
  // asymmetric props so opposite sides of the orbit look different, like a real room
  body.rotation.y = 0.5;
  const props = [
    [new THREE.ConeGeometry(0.06, 0.2, 24), '#2a2', [0.35, 0.1, -0.2]],
    [new THREE.TorusKnotGeometry(0.05, 0.018, 80, 12), '#fa0', [-0.3, 0.07, 0.25]],
    [new THREE.BoxGeometry(1.2, 0.9, 0.05), '#245', [0, 0.45, -1.6]],
    [new THREE.CylinderGeometry(0.25, 0.25, 1.4, 24), '#a35', [1.7, 0.7, 0.6]],
  ];
  for (const [geo, base, at] of props) { const m = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ map: tex(base, 60, 256) })); m.position.set(...at); scene.add(m); }
  const cam = new THREE.PerspectiveCamera(55, 960 / 720, 0.01, 50);
  const files = [];
  for (const [r, h] of [[0.62, 0.22], [0.55, 0.48]]) for (let i = 0; i < 22; i++) {
    const a = (i / 22) * Math.PI * 2 + (h > 0.3 ? 0.14 : 0);
    cam.position.set(r * Math.sin(a), h, r * Math.cos(a)); cam.lookAt(0, 0.17, 0);
    renderer.render(scene, cam);
    const blob = await new Promise((res) => renderer.domElement.toBlob(res, 'image/jpeg', 0.93));
    files.push(new File([blob], `orbit_${String(files.length).padStart(2, '0')}.jpg`, { type: 'image/jpeg' }));
  }
  window.__orbitFiles = files;
  const dt = new DataTransfer(); files.forEach((f) => dt.items.add(f));
  const input = document.querySelector('#scan-file-input'); input.files = dt.files; input.dispatchEvent(new Event('change'));
  window.__t0 = performance.now();
  if (build) document.querySelector('#scan-build-button').click();
  return files.length;
};
window.scanStatus = () => ({
  t: Math.round((performance.now() - window.__t0) / 1000),
  title: document.querySelector('#scan-stage-title').textContent,
  pct: document.querySelector('#scan-percentage').textContent,
  detail: document.querySelector('#scan-detail').textContent,
  stats: document.querySelector('#scan-mesh-stats').textContent,
  err: document.querySelector('#scan-error').hidden ? null : document.querySelector('#scan-error').textContent,
});
window.loadSaved = () => new Promise((res) => { const r = indexedDB.open('scan-dev', 1); r.onupgradeneeded = () => r.result.createObjectStore('k'); r.onsuccess = () => { const g = r.result.transaction('k').objectStore('k').get('last'); g.onsuccess = () => res(g.result); }; });
window.saveScan = () => new Promise((res, rej) => { const s = window.spatialDropScan; const ses = s.session; const names = ses.recon.cams.map((k) => ses.frames[k.imgIdx].name); const r = indexedDB.open('scan-dev', 1); r.onupgradeneeded = () => r.result.createObjectStore('k'); r.onsuccess = () => { const tx = r.result.transaction('k', 'readwrite'); tx.objectStore('k').put({ data: s.splat.data, n: s.splat.n, dc: s.splat.dc, cameras: s.cameras, viewMetas: s.viewMetas, names }, 'last'); tx.oncomplete = () => res(true); tx.onerror = rej; }; });

// Signed distance to the true scene (meters, +Y up, floor at 0), matching a mesh built with heightMeters: 0.41.
const sdfBox = (p) => {
  const c = Math.cos(-0.5), s = Math.sin(-0.5);
  const x = c * p[0] + s * p[2], z = -s * p[0] + c * p[2], y = p[1] - 0.13;
  const q = [Math.abs(x) - 0.09, Math.abs(y) - 0.13, Math.abs(z) - 0.07];
  return Math.hypot(Math.max(q[0], 0), Math.max(q[1], 0), Math.max(q[2], 0)) + Math.min(Math.max(q[0], q[1], q[2]), 0);
};
const groundTruthSdf = (p) => Math.min(sdfBox(p), Math.hypot(p[0], p[1] - 0.33, p[2]) - 0.08);

/** Shape error of a scan mesh against the known scene, after the best turn and shift about +Y. */
window.shapeError = (mesh) => {
  const P = mesh.positions, V = P.length / 3;
  const cost = (rot, dx, dz) => {
    const c = Math.cos(rot), s = Math.sin(rot); let sum = 0;
    for (let v = 0; v < V; v += 13) { const x = P[v * 3], z = P[v * 3 + 2]; sum += Math.min(0.02, Math.abs(groundTruthSdf([c * x + s * z + dx, P[v * 3 + 1], -s * x + c * z + dz]))); }
    return sum;
  };
  let best = { c: Infinity, rot: 0, dx: 0, dz: 0 };
  for (let rot = -Math.PI; rot < Math.PI; rot += 0.04) { const c = cost(rot, 0, 0); if (c < best.c) best = { c, rot, dx: 0, dz: 0 }; }
  for (let it = 0; it < 3; it++) {
    const r0 = best.rot;
    for (let rot = r0 - 0.04; rot <= r0 + 0.04; rot += 0.01) for (let dx = -0.03; dx <= 0.03; dx += 0.005) for (let dz = -0.03; dz <= 0.03; dz += 0.005) {
      const c = cost(rot, dx, dz); if (c < best.c) best = { c, rot, dx, dz };
    }
  }
  const c = Math.cos(best.rot), s = Math.sin(best.rot); const d = [];
  for (let v = 0; v < V; v++) { const x = P[v * 3], z = P[v * 3 + 2]; d.push(groundTruthSdf([c * x + s * z + best.dx, P[v * 3 + 1], -s * x + c * z + best.dz])); }
  const abs = d.map(Math.abs).sort((a, b) => a - b);
  return {
    meanMm: +(abs.reduce((a, b) => a + b, 0) / abs.length * 1000).toFixed(2),
    medianMm: +(abs[abs.length >> 1] * 1000).toFixed(1),
    p95Mm: +(abs[Math.floor(abs.length * 0.95)] * 1000).toFixed(1),
    dentsOver4mm: +(d.filter((x) => x < -0.004).length / V * 100).toFixed(1),
    bulgesOver4mm: +(d.filter((x) => x > 0.004).length / V * 100).toFixed(1),
  };
};
/** Camera models for buildScanMesh's `views` option from saved view metadata. */
window.cameraModels = (metas) => metas.map((m) => ({ R: m.R, t: m.t, fx: m.f, fy: m.fy ?? m.f, cx: m.cx ?? m.featureWidth / 2, cy: m.cy ?? m.featureHeight / 2, width: m.featureWidth, height: m.featureHeight }));

/** Overlay horizontal slices of meshes (red) on the true outline (green): { label: mesh } */
window.crossSections = (meshes, heights = [[0.13, 'box mid'], [0.33, 'sphere mid']]) => {
  document.getElementById('cmp')?.remove();
  const box = [[-0.09, -0.07], [0.09, -0.07], [0.09, 0.07], [-0.09, 0.07]].map(([x, z]) => [Math.cos(0.5) * x + Math.sin(0.5) * z, -Math.sin(0.5) * x + Math.cos(0.5) * z]);
  const S = (v) => (v * 1100 + 160).toFixed(1);
  let html = '';
  for (const [name, mesh] of Object.entries(meshes)) {
    const P = mesh.positions;
    let best = null;
    for (let rot = -Math.PI; rot < Math.PI; rot += 0.02) {
      const c = Math.cos(rot), s = Math.sin(rot); let sum = 0;
      for (let v = 0; v < P.length / 3; v += 17) {
        if (Math.abs(P[v * 3 + 1] - 0.13) > 0.05) continue;
        const X = c * P[v * 3] + s * P[v * 3 + 2], Z = -s * P[v * 3] + c * P[v * 3 + 2];
        const bx = Math.cos(-0.5) * X + Math.sin(-0.5) * Z, bz = -Math.sin(-0.5) * X + Math.cos(-0.5) * Z;
        sum += Math.abs(Math.max(Math.abs(bx) - 0.09, Math.abs(bz) - 0.07));
      }
      if (!best || sum < best.sum) best = { sum, rot };
    }
    const c = Math.cos(best.rot), s = Math.sin(best.rot);
    for (const [h, label] of heights) {
      const pts = [];
      for (let v = 0; v < P.length / 3; v++) if (Math.abs(P[v * 3 + 1] - h) <= 0.004) pts.push([c * P[v * 3] + s * P[v * 3 + 2], -s * P[v * 3] + c * P[v * 3 + 2]]);
      const truth = h < 0.2 ? `<polygon points="${box.map(([x, z]) => `${S(x)},${S(z)}`).join(' ')}" fill="none" stroke="#0a0" stroke-width="2"/>` : `<circle cx="${S(0)}" cy="${S(0)}" r="88" fill="none" stroke="#0a0" stroke-width="2"/>`;
      html += `<figure style="margin:0;color:#fff;font:11px monospace"><svg viewBox="0 0 320 320" style="width:100%;background:#fff">${truth}${pts.map(([x, z]) => `<circle cx="${S(x)}" cy="${S(z)}" r="1.3" fill="#c00"/>`).join('')}</svg><br>${name} · ${label}</figure>`;
    }
  }
  const wrap = document.createElement('div'); wrap.id = 'cmp';
  wrap.style.cssText = 'position:fixed;inset:0;z-index:99999;background:#111;display:grid;grid-template-columns:1fr 1fr;gap:6px;padding:6px;overflow:auto';
  wrap.innerHTML = html; document.body.append(wrap);
};

/** The crossSections overlay as a PNG data URL (works while the tab is hidden). */
window.crossSectionPng = (meshes, heights = [0.06, 0.13, 0.22, 0.33]) => {
  const box = [[-0.09, -0.07], [0.09, -0.07], [0.09, 0.07], [-0.09, 0.07]].map(([x, z]) => [Math.cos(0.5) * x + Math.sin(0.5) * z, -Math.sin(0.5) * x + Math.cos(0.5) * z]);
  const cell = 220, names = Object.keys(meshes);
  const canvas = document.createElement('canvas'); canvas.width = cell * heights.length; canvas.height = cell * names.length;
  const g = canvas.getContext('2d'); g.fillStyle = '#fff'; g.fillRect(0, 0, canvas.width, canvas.height);
  names.forEach((name, row) => {
    const P = meshes[name].positions; let best = null;
    for (let rot = -Math.PI; rot < Math.PI; rot += 0.02) {
      const c = Math.cos(rot), s = Math.sin(rot); let sum = 0;
      for (let v = 0; v < P.length / 3; v += 17) {
        if (Math.abs(P[v * 3 + 1] - 0.13) > 0.05) continue;
        const X = c * P[v * 3] + s * P[v * 3 + 2], Z = -s * P[v * 3] + c * P[v * 3 + 2];
        const bx = Math.cos(-0.5) * X + Math.sin(-0.5) * Z, bz = -Math.sin(-0.5) * X + Math.cos(-0.5) * Z;
        sum += Math.abs(Math.max(Math.abs(bx) - 0.09, Math.abs(bz) - 0.07));
      }
      if (!best || sum < best.sum) best = { sum, rot };
    }
    const c = Math.cos(best.rot), s = Math.sin(best.rot);
    heights.forEach((h, col) => {
      const ox = col * cell + cell / 2, oy = row * cell + cell / 2, k = 700;
      g.strokeStyle = '#0a0'; g.lineWidth = 2; g.beginPath();
      if (h < 0.26) { box.forEach(([x, z], i) => (i ? g.lineTo : g.moveTo).call(g, ox + x * k, oy + z * k)); g.closePath(); }
      else g.arc(ox, oy, Math.sqrt(Math.max(0, 0.08 ** 2 - (h - 0.33) ** 2)) * k, 0, Math.PI * 2);
      g.stroke(); g.fillStyle = '#c00';
      for (let v = 0; v < P.length / 3; v++) {
        if (Math.abs(P[v * 3 + 1] - h) > 0.003) continue;
        g.fillRect(ox + (c * P[v * 3] + s * P[v * 3 + 2]) * k - 1, oy + (-s * P[v * 3] + c * P[v * 3 + 2]) * k - 1, 2, 2);
      }
      g.fillStyle = '#000'; g.font = '11px monospace'; g.fillText(`${name} y=${h}`, col * cell + 4, row * cell + 12);
    });
  });
  return canvas.toDataURL('image/png');
};
