import React, { useState, useRef } from 'react';
import { Upload, Eye, EyeOff, AlertTriangle, CheckCircle, Zap } from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || "/api";

export default function DragDropUpload() {
  const [dragActive, setDragActive] = useState(false);
  const [image, setImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [includeGradCam, setIncludeGradCam] = useState(true);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current.click();
  };

  const processFile = (file) => {
    if (!file.type.startsWith('image/')) {
      setError("Please upload an image file (.jpg, .png, .bmp)");
      return;
    }
    setError(null);
    setImage(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
  };

  const runPrediction = async () => {
    if (!image) return;
    setLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append("file", image);
    
    try {
      const response = await fetch(`${API_URL}/predict?heatmap=${includeGradCam}`, {
        method: "POST",
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      setError("Failed to run prediction. Verify that the backend server is running on port 8000.");
    } finally {
      setLoading(false);
    }
  };

  const clearUpload = () => {
    setImage(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
  };

  const isCrack = result && result.prediction === "Crack Detected";

  return (
    <div className="space-y-6">
      <div className="flex flex-col lg:flex-row gap-6">
        {/* Upload Panel */}
        <div className="flex-1 glass-panel rounded-2xl p-6 glow-box-blue flex flex-col justify-between min-h-[350px]">
          <div>
            <h2 className="text-xl font-bold mb-4 tracking-tight flex items-center gap-2">
              <Upload className="text-sky-400 w-5 h-5" /> File Upload & Preprocessing
            </h2>
            <p className="text-slate-400 text-sm mb-6">
              Drag and drop an image of concrete or masonry here to scan for defects.
            </p>
          </div>

          {!previewUrl ? (
            <div 
              className={`flex-1 border-2 border-dashed rounded-xl flex flex-col items-center justify-center p-8 cursor-pointer transition-all duration-300 ${
                dragActive ? 'border-sky-400 bg-sky-950/20 scale-[0.99]' : 'border-slate-800 hover:border-slate-700 bg-slate-900/10'
              }`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              onClick={triggerFileInput}
            >
              <input 
                ref={fileInputRef} 
                type="file" 
                className="hidden" 
                onChange={handleChange} 
                accept="image/*"
              />
              <div className="p-4 bg-slate-900/50 rounded-full mb-4 border border-slate-800">
                <Upload className="w-8 h-8 text-sky-400 animate-pulse" />
              </div>
              <p className="font-medium text-slate-200">Drag and drop file here</p>
              <p className="text-xs text-slate-500 mt-2">Supports JPG, JPEG, PNG, BMP up to 10MB</p>
            </div>
          ) : (
            <div className="flex-1 flex flex-col items-center justify-center gap-4">
              <div className="relative max-h-[250px] overflow-hidden rounded-xl border border-slate-800 bg-slate-900">
                <img 
                  src={previewUrl} 
                  alt="Preview" 
                  className="max-h-[250px] w-auto object-contain"
                />
              </div>
              <p className="text-xs text-slate-400 truncate max-w-[250px]">{image.name}</p>
            </div>
          )}

          <div className="mt-6 flex flex-col sm:flex-row gap-3">
            {previewUrl && (
              <>
                <button
                  onClick={runPrediction}
                  disabled={loading}
                  className="flex-1 py-3 px-4 bg-sky-600 hover:bg-sky-500 disabled:bg-sky-800 font-semibold text-white rounded-xl shadow-lg hover:shadow-sky-500/20 transition-all flex items-center justify-center gap-2"
                >
                  {loading ? (
                    <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  ) : (
                    <>
                      <Zap className="w-5 h-5 fill-white" /> Analyze Image
                    </>
                  )}
                </button>
                <button
                  onClick={clearUpload}
                  disabled={loading}
                  className="py-3 px-4 bg-slate-900 hover:bg-slate-800 border border-slate-800 font-semibold text-slate-300 rounded-xl transition-all"
                >
                  Clear
                </button>
              </>
            )}
          </div>
        </div>

        {/* Inference & Heatmap Results Panel */}
        <div className="flex-1 glass-panel rounded-2xl p-6 glow-box-blue flex flex-col">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-bold tracking-tight">Analysis Results</h2>
            {result && result.heatmap_image && (
              <button 
                onClick={() => setShowHeatmap(!showHeatmap)}
                className="flex items-center gap-1.5 text-xs text-sky-400 hover:text-sky-300 border border-sky-400/20 bg-sky-950/20 px-2.5 py-1.5 rounded-lg transition-all"
              >
                {showHeatmap ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                {showHeatmap ? "Hide Heatmap" : "Show Heatmap"}
              </button>
            )}
          </div>

          {error && (
            <div className="flex-1 bg-red-950/20 border border-red-800/30 text-red-300 p-4 rounded-xl flex items-start gap-3 my-auto">
              <AlertTriangle className="w-5 h-5 text-red-500 shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold">Inference Error</p>
                <p className="text-sm text-red-400/80">{error}</p>
              </div>
            </div>
          )}

          {!result && !loading && !error && (
            <div className="flex-1 flex flex-col items-center justify-center text-slate-500 p-8 border border-dashed border-slate-800/50 rounded-xl my-auto">
              <Eye className="w-12 h-12 text-slate-700 mb-3" />
              <p className="text-sm">Upload an image and click "Analyze" to see predictions</p>
            </div>
          )}

          {loading && (
            <div className="flex-1 flex flex-col items-center justify-center text-slate-400 p-8 my-auto space-y-4">
              <div className="relative w-16 h-16">
                <div className="absolute inset-0 rounded-full border-4 border-slate-800"></div>
                <div className="absolute inset-0 rounded-full border-4 border-sky-500 border-t-transparent animate-spin"></div>
              </div>
              <div className="text-center">
                <p className="font-semibold text-slate-200 animate-pulse">Running Neural Networks...</p>
                <p className="text-xs text-slate-500 mt-1">Applying Grad-CAM backpropagation</p>
              </div>
            </div>
          )}

          {result && !loading && !error && (
            <div className="flex-1 flex flex-col justify-between space-y-6">
              {/* Output Indicator */}
              <div className={`p-4 rounded-xl border flex items-center gap-4 ${
                isCrack 
                  ? 'bg-red-950/20 border-red-800/40 text-red-200' 
                  : 'bg-emerald-950/20 border-emerald-800/40 text-emerald-200'
              }`}>
                {isCrack ? (
                  <AlertTriangle className="w-8 h-8 text-red-500 shrink-0" />
                ) : (
                  <CheckCircle className="w-8 h-8 text-emerald-500 shrink-0" />
                )}
                <div>
                  <h3 className="text-lg font-bold uppercase tracking-wide">
                    {result.prediction}
                  </h3>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Confidence: {(result.confidence * 100).toFixed(1)}% | Latency: {result.inference_time_ms.toFixed(1)}ms
                  </p>
                </div>
              </div>

              {/* Visualization Frame */}
              {result.heatmap_image && showHeatmap ? (
                <div className="relative flex-1 max-h-[250px] overflow-hidden rounded-xl border border-slate-800 bg-slate-900 flex justify-center">
                  <img 
                    src={result.heatmap_image} 
                    alt="GradCAM Heatmap Overlay" 
                    className="max-h-[250px] w-auto object-contain"
                  />
                  <div className="absolute bottom-2 left-2 bg-black/60 backdrop-blur-md px-2.5 py-1 rounded text-[10px] text-slate-400 font-semibold border border-white/5">
                    Grad-CAM Thermal Defect Map
                  </div>
                </div>
              ) : (
                <div className="relative flex-1 max-h-[250px] overflow-hidden rounded-xl border border-slate-800 bg-slate-900 flex justify-center">
                  <img 
                    src={previewUrl} 
                    alt="Original Image" 
                    className="max-h-[250px] w-auto object-contain"
                  />
                  <div className="absolute bottom-2 left-2 bg-black/60 backdrop-blur-md px-2.5 py-1 rounded text-[10px] text-slate-400 font-semibold border border-white/5">
                    Original Scan View
                  </div>
                </div>
              )}

              {/* Progress/Confidence Bar */}
              <div className="space-y-2">
                <div className="flex justify-between text-xs font-semibold text-slate-400">
                  <span>Crack Probability</span>
                  <span className={isCrack ? 'text-red-400' : 'text-emerald-400'}>
                    {(result.probability * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-slate-900 rounded-full h-3 border border-slate-800 overflow-hidden">
                  <div 
                    className={`h-full rounded-full transition-all duration-500 ${isCrack ? 'bg-gradient-to-r from-red-600 to-orange-500' : 'bg-gradient-to-r from-emerald-600 to-teal-500'}`}
                    style={{ width: `${result.probability * 100}%` }}
                  ></div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Grad-CAM settings options */}
      <div className="glass-panel rounded-2xl p-4 flex flex-row items-center justify-between text-sm text-slate-300">
        <div className="flex items-center gap-2">
          <input 
            type="checkbox" 
            id="gradcam_opt" 
            checked={includeGradCam} 
            onChange={(e) => setIncludeGradCam(e.target.checked)}
            className="w-4 h-4 rounded text-sky-500 bg-slate-950 border-slate-800 focus:ring-sky-500"
          />
          <label htmlFor="gradcam_opt" className="cursor-pointer font-medium select-none">
            Generate Grad-CAM heatmaps during single prediction
          </label>
        </div>
        <span className="text-xs text-slate-500 hidden md:inline">
          Disable to speed up inference response (by skipping backprop)
        </span>
      </div>
    </div>
  );
}
