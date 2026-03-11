import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App.tsx";
import { USE_FLOAT32 } from "./numbl-core/runtime/types.ts";

if (USE_FLOAT32) {
  console.info("Using Float32 precision for tensors (NUMBL_USE_FLOAT32=true)");
} else {
  console.info("Using Float64 precision for tensors (NUMBL_USE_FLOAT32=false)");
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
