import React, { useState, useRef } from 'react';
import { Layers, FileText, CheckCircle, AlertTriangle, AlertCircle, RefreshCw } from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || "/api";

export default function BatchUpload() {
  const [files, setFiles] = useState([]);
  const [previews, setPreviews] = useState({});
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [progress, setProgress] = useState(0);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const selectedFiles = Array.from(e.target.files);
      setFiles(selectedFiles);
      setResults(null);
      setError(null);

      // Generate object URL previews for thumbnails
      const newPreviews = {};
      selectedFiles.forEach((file) => {
        if (file.type.startsWith('image/')) {
          newPreviews[file.name] = URL.createObjectURL(file);
        }
      });
      setPreviews(newPreviews);
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current.click();
  };

  const uploadBatch = async () => {
    if (files.length === 0) return;
    setLoading(true);
    setError(null);
    setProgress(10);

    const CHUNK_SIZE = 10;
    const totalFiles = files.length;
    const chunks = [];
    for (let i = 0; i < totalFiles; i += CHUNK_SIZE) {
      chunks.push(files.slice(i, i + CHUNK_SIZE));
    }

    const allPredictions = [];
    let totalLatencySum = 0;
    let successfulImagesCount = 0;

    try {
      for (let i = 0; i < chunks.length; i++) {
        const chunk = chunks[i];
        const formData = new FormData();
        chunk.forEach((file) => {
          formData.append("files", file);
        });

        const response = await fetch(`${API_URL}/predict-batch`, {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        allPredictions.push(...data.predictions);
        totalLatencySum += data.average_latency_ms * data.total_images;
        successfulImagesCount += data.total_images;

        const percentComplete = Math.round(10 + ((i + 1) / chunks.length) * 90);
        setProgress(percentComplete);
      }

      setResults({
        total_images: successfulImagesCount,
        predictions: allPredictions,
        average_latency_ms: successfulImagesCount > 0 ? (totalLatencySum / successfulImagesCount) : 0
      });
    } catch (err) {
      console.error(err);
      setError("Failed to run batch predictions. Verify backend server connectivity.");
    } finally {
      setLoading(false);
    }
  };

  const clearBatch = () => {
    setFiles([]);
    setPreviews({});
    setResults(null);
    setError(null);
    setProgress(0);
  };

  // Compute summary stats
  const total = results ? results.total_images : 0;
  const cracks = results ? results.predictions.filter(p => p.prediction === "Crack Detected").length : 0;
  const healthy = total - cracks;
  const avgLatency = results ? results.average_latency_ms : 0;

  return (
    <div className="space-y-6">
      {/* Batch Control Card */}
      <div className="glass-panel rounded-2xl p-6 glow-box-blue">
        <h2 className="text-xl font-bold mb-4 tracking-tight flex items-center gap-2">
          <Layers className="text-sky-400 w-5 h-5" /> Batch File Processing
        </h2>
        <p className="text-slate-400 text-sm mb-6">
          Upload up to 100 images simultaneously for rapid batched concrete inspection.
        </p>

        <div className="flex flex-col md:flex-row gap-4 items-center justify-between">
          <div className="flex gap-3 w-full md:w-auto">
            <button
              onClick={triggerFileInput}
              disabled={loading}
              className="flex-1 md:flex-none py-2.5 px-5 bg-slate-900 hover:bg-slate-800 border border-slate-800 font-semibold text-slate-300 rounded-xl transition-all"
            >
              Select Images
            </button>
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept="image/*"
              className="hidden"
              onChange={handleFileChange}
            />

            {files.length > 0 && (
              <button
                onClick={uploadBatch}
                disabled={loading}
                className="flex-1 md:flex-none py-2.5 px-6 bg-sky-600 hover:bg-sky-500 font-semibold text-white rounded-xl shadow-lg transition-all flex items-center justify-center gap-2"
              >
                {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : "Run Batch Predict"}
              </button>
            )}

            {files.length > 0 && (
              <button
                onClick={clearBatch}
                disabled={loading}
                className="py-2.5 px-4 bg-slate-950 text-slate-500 hover:text-slate-400 border border-slate-900 rounded-xl transition-all"
              >
                Clear
              </button>
            )}
          </div>
          <span className="text-xs text-slate-500 font-semibold uppercase tracking-wider">
            {files.length > 0 ? `${files.length} images staged` : "No files staged"}
          </span>
        </div>

        {/* Progress Bar */}
        {loading && (
          <div className="mt-6 space-y-2">
            <div className="flex justify-between text-xs text-slate-400 font-semibold">
              <span>Uploading and analyzing batched tensors...</span>
              <span>{progress}%</span>
            </div>
            <div className="w-full bg-slate-950 rounded-full h-2 border border-slate-900 overflow-hidden">
              <div
                className="h-full bg-sky-500 rounded-full transition-all duration-300"
                style={{ width: `${progress}%` }}
              ></div>
            </div>
          </div>
        )}

        {error && (
          <div className="mt-4 bg-red-950/20 border border-red-800/30 text-red-300 p-4 rounded-xl flex items-center gap-3">
            <AlertCircle className="w-5 h-5 text-red-500" />
            <p className="text-sm">{error}</p>
          </div>
        )}
      </div>

      {/* Summary Diagnostics Metrics */}
      {results && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="glass-card rounded-xl p-4 flex flex-col justify-center">
            <span className="text-xs text-slate-500 font-semibold uppercase tracking-wider">Total Scans</span>
            <span className="text-2xl font-bold text-sky-400 mt-1">{total}</span>
          </div>
          <div className="glass-card rounded-xl p-4 flex flex-col justify-center">
            <span className="text-xs text-slate-500 font-semibold uppercase tracking-wider">Cracks Found</span>
            <span className="text-2xl font-bold text-red-500 mt-1">{cracks}</span>
          </div>
          <div className="glass-card rounded-xl p-4 flex flex-col justify-center">
            <span className="text-xs text-slate-500 font-semibold uppercase tracking-wider">Healthy Elements</span>
            <span className="text-2xl font-bold text-emerald-500 mt-1">{healthy}</span>
          </div>
          <div className="glass-card rounded-xl p-4 flex flex-col justify-center">
            <span className="text-xs text-slate-500 font-semibold uppercase tracking-wider">Avg Latency</span>
            <span className="text-2xl font-bold text-amber-500 mt-1">{avgLatency.toFixed(1)} ms</span>
          </div>
        </div>
      )}

      {/* Table Results */}
      {files.length > 0 && (
        <div className="glass-panel rounded-2xl overflow-hidden border border-slate-800">
          <div className="max-h-[500px] overflow-y-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-slate-900 border-b border-slate-800 text-xs text-slate-400 font-semibold uppercase tracking-wider">
                  <th className="py-4 px-6">Preview</th>
                  <th className="py-4 px-6">Filename</th>
                  <th className="py-4 px-6 text-center">Prediction</th>
                  <th className="py-4 px-6 text-right">Confidence</th>
                  <th className="py-4 px-6 text-right">Latency</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-900">
                {files.map((file, idx) => {
                  const name = file.name;
                  const preview = previews[name];
                  const predResult = results ? results.predictions.find(p => p.filename === name) : null;
                  const isCrackFound = predResult && predResult.prediction === "Crack Detected";

                  return (
                    <tr key={idx} className="hover:bg-slate-900/40 transition-colors">
                      <td className="py-3 px-6">
                        <div className="w-12 h-12 rounded-lg bg-slate-900 border border-slate-800 overflow-hidden flex items-center justify-center">
                          {preview ? (
                            <img src={preview} alt="Thumb" className="object-cover w-full h-full" />
                          ) : (
                            <FileText className="w-5 h-5 text-slate-700" />
                          )}
                        </div>
                      </td>
                      <td className="py-3 px-6 text-sm font-medium text-slate-200 max-w-[200px] truncate">
                        {name}
                      </td>
                      <td className="py-3 px-6 text-center">
                        {predResult ? (
                          <div className={`inline-flex items-center gap-1 px-3 py-1 rounded-full text-xs font-semibold uppercase border ${isCrackFound
                              ? 'bg-red-950/20 border-red-800/30 text-red-400'
                              : 'bg-emerald-950/20 border-emerald-800/30 text-emerald-400'
                            }`}>
                            {isCrackFound ? <AlertTriangle className="w-3.5 h-3.5" /> : <CheckCircle className="w-3.5 h-3.5" />}
                            {predResult.prediction}
                          </div>
                        ) : (
                          <span className="text-slate-500 text-xs italic">Awaiting calculation</span>
                        )}
                      </td>
                      <td className="py-3 px-6 text-right text-sm font-semibold text-slate-300">
                        {predResult ? `${(predResult.confidence * 100).toFixed(1)}%` : "-"}
                      </td>
                      <td className="py-3 px-6 text-right text-sm font-mono text-slate-500">
                        {predResult ? `${predResult.inference_time_ms.toFixed(1)}ms` : "-"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
