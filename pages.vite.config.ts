import { defineConfig } from "vite";

const repositoryName = process.env.GITHUB_REPOSITORY?.split("/")[1];

export default defineConfig({
  root: "pages-src",
  base: repositoryName ? `/${repositoryName}/` : "/",
  build: {
    outDir: "../pages-dist",
    emptyOutDir: true,
  },
});
