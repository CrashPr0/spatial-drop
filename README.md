# Spatial Drop

Spatial Drop turns a GLB model into an AR-ready web experience. The full application supports persistent uploads, shareable links, QR codes, Apple Quick Look, Android Scene Viewer, and WebXR.

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
```

- `npm run build` builds the full application and upload API for a Cloudflare-compatible runtime.
- `npm run build:pages` builds the static GitHub Pages preview.

## GitHub Pages preview

The workflow in `.github/workflows/pages.yml` deploys `pages-dist` whenever `main` changes. Because GitHub Pages is static, its version can preview a local GLB or generate a shareable AR link from an existing public HTTPS GLB URL. It cannot accept persistent runtime uploads by itself.

To restore one-click uploads on the public site, connect the Pages frontend to the included upload API deployed on a serverless host with object storage.
