import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  root: "src/plot-viewer",
  base: "/",
  build: {
    outDir: "../../dist-plot-viewer",
    emptyOutDir: true,
  },
});
