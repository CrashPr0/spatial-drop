# Spatial Drop

Spatial Drop lets you preview, share, and place 3D models in AR, with experimental on-device scanning in the Pages preview.

## What you can do

- View a GLB, rotate it, and zoom in. Choose floor or wall placement and whether to allow resizing. The full app's publisher at `/publish` uploads models up to 15 MB and creates a persistent share link and QR code. In the Pages **Studio**, files stay local; cross-device share links require a public HTTPS GLB URL.
- Tap **View in your space** for Apple Quick Look on iOS, Scene Viewer on Android, or WebXR on supported devices and browsers. Local scans also offer **Place with 8th Wall** for compatible mobile browsers.
- Use **Scan** in the Pages preview to capture an object without uploading photos. A three-second countdown gives you time to get ready. The direction arrow guides your orbit; hold still while the ring fills to have photos kept automatically. If you move too far, the arrow points back and a faded last photo helps you line up again.
- Reconstruct a scan locally with WebGPU, adjust its height, crop, density, and detail, then preview it or download GLB, USDZ, or splat files. You can also choose existing photos or one video. The full app's home page instead previews existing Apple USDZ scans or GLBs locally.
- Use **QR Beam** and **Receive** in the Pages preview to transfer a compact GLB between a screen and a phone camera through animated QR or Cimbar frames.

## Workshop quickstart

These steps use the Pages preview, which has **Studio**, **Scan**, **QR Beam**, and **Receive** tabs. The full app has a different home screen; use `/publish` there for uploads and sharing.

1. Open the site on your phone and select **Studio**.
2. Tap **Drop a GLB here** and choose a `.glb` file up to 15 MB. Drag the model to look around it. Choose **Floor** or **Wall** and whether to allow pinch resizing.
3. On a supported phone, tap **View in your space**. Move your phone to find a surface, then place the model. For a cross-device AR link, use a public model URL as described next.
4. To share, enter a publicly accessible HTTPS URL ending in `.glb` in **Public GLB URL**, add a display name, and tap **Create share link**. Another participant can scan the displayed QR with their phone camera, or you can tap **Copy link**. A local file alone does not create a cross-device link. In the full app, choose your GLB at `/publish` and tap **Publish AR page** to get the link and QR.
5. Select **Scan**. Put an object on a textured table in even light, tap **Start capture**, and allow camera access. After the countdown, circle slowly at arm's length, following the arrow. Pause when told to hold; photos are kept automatically. Try a second, higher loop. Capture at least 12 photos (30–60 recommended), tap **Stop capture**, then **Build 3D model**. Keep the tab in front while it works; a phone can take several minutes. When ready, view the result in AR or tap **Download GLB**.

### Troubleshooting

- **File rejected or upload failed:** choose a valid, nonempty `.glb` within the publisher/Studio's 15 MB limit; export again or reduce its size if needed. If publishing fails, check your connection and retry.
- **Share link unavailable:** the Pages preview needs a public HTTPS URL pointing to a `.glb` file. Local previews and scans stay on your device.
- **Copy link fails:** copy the displayed link manually; the app tries selecting it when automatic copying is unavailable.
- **Camera blocked or unavailable:** allow camera access, or choose photos or a video in Scan. For 8th Wall AR errors, check camera and motion permissions and use a compatible iPhone or Android browser.
- **Scan cannot build:** scanning needs WebGPU. The app suggests Safari 26 or newer on iPhone, or recent Chrome/Edge. Keep the tab in front if the GPU is reclaimed; if too few photos can be placed, rescan slowly with more overlap.

## Development

Requires Node.js 22.13 or newer.

```bash
npm install
npm run dev
```

The full application runs at `http://localhost:3000` and uses a locally simulated R2 bucket for uploaded models.

## Builds

```bash
npm run lint
npm run build
npm run build:pages
npm run test
npm run test:beam
npm run test:scan
```

- `npm run build` builds the full application and upload API for a Cloudflare-compatible runtime.
- `npm run build:pages` builds the static GitHub Pages preview.
- `npm run test` builds the full app, then runs the rendered-HTML tests.
- `npm run test:beam` tests the QR Beam codec.
- `npm run test:scan` tests scan meshing, USDZ export, texturing, cropping, and capture guidance.

## GitHub Pages preview

The workflow in `.github/workflows/pages.yml` deploys `pages-dist` whenever `main` changes. Because GitHub Pages is static, its version can preview a local GLB or generate a shareable AR link from an existing public HTTPS GLB URL. It cannot accept persistent runtime uploads by itself.

To restore one-click uploads on the public site, connect the Pages frontend to the included upload API deployed on a serverless host with object storage.
