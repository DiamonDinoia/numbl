import { defineConfig } from "vitest/config";

/**
 * Vitest config for the LAPACK native addon tests.
 * Run with: npm run test:lapack
 */
export default defineConfig({
  define: {
    "import.meta.env.NUMBL_USE_FLOAT32": JSON.stringify("false"),
    "import.meta.env.NUMBL_DISABLE_WEBGPU": JSON.stringify("true"),
  },
  test: {
    include: ["src/numbl-core/__tests__/native/**/*.test.ts"],
  },
});
