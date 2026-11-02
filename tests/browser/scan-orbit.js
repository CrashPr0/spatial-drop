// Synthetic object-orbit capture for exercising Scan mode end to end.
// With the Pages dev server running (npx vite --config pages.vite.config.ts), in the page console:
//   await import('/@fs' + '<absolute repo path>/tests/browser/scan-orbit.js');
//   await makeOrbit();            // renders 44 views of a known scene and starts a build
//   scanStatus();                 // progress
//   await saveScan();             // keep the trained splat in IndexedDB
//   await spatialDropLoadSplat(await loadSaved());   // re-mesh it after editing scan-mesh.ts
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
window.saveScan = () => new Promise((res, rej) => { const s = window.spatialDropScan; const ses = s.session; const names = ses.recon.cams.map((k) => ses.frames[k.imgIdx].name); const r = indexedDB.open('scan-dev', 1); r.onupgradeneeded = () => r.result.createObjectStore('k'); r.onsuccess = () => { const tx = r.result.transaction('k', 'readwrite'); tx.objectStore('k').put({ data: s.splat.data, n: s.splat.n, dc: s.splat.dc, cameras: s.cameras, names }, 'last'); tx.oncomplete = () => res(true); tx.onerror = rej; }; });
