import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { PlotViewerApp } from "./PlotViewerApp";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <PlotViewerApp />
  </StrictMode>
);
