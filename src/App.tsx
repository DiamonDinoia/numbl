import { BrowserRouter, Routes, Route } from "react-router-dom";
import { ProjectListPage } from "./pages/ProjectListPage";
import { ProjectIDEPage } from "./pages/ProjectIDEPage";
import { ShareIDEPage } from "./pages/ShareIDEPage";

function App() {
  console.log("App initialized");
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<ProjectListPage />} />
        <Route path="/project/:projectName" element={<ProjectIDEPage />} />
        <Route path="/share" element={<ShareIDEPage />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
