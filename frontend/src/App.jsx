import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Sidebar from "./components/Sidebar";
import Dashboard from "./pages/Dashboard";
import CryAnalysis from "./pages/CryAnalysis";
import Health from "./pages/Health";
import UploadAudio from "./pages/UploadAudio";
import CryAnalysisDashboard from "./pages/CryAnalysisDashboard";
import UploadAudioWithGradCAM from "./pages/UploadAudioWithGradCAM";
import CryAnalysisFull from "./pages/CryAnalysisFull";

function App() {
  return (
    <Router>
      <div className="d-flex">
        <Sidebar />
        <div className="flex-grow-1">
          <Navbar />
          <div className="container mt-4">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/cry-analysis" element={<CryAnalysis />} />
              <Route path="/health" element={<Health />} />
              <Route path="/cry-analysis" element={<CryAnalysisDashboard />} />
              <Route path="/upload-gradcam" element={<UploadAudioWithGradCAM />} />
              <Route path="/cry-analysis-full" element={<CryAnalysisFull />} />
              <Route path="/upload-audio" element={<UploadAudio />} />
            </Routes>
          </div>
        </div>
      </div>
    </Router>
  );
}
export default App;
