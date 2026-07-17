import React, { useState, useEffect } from 'react';
import { Cpu, CheckCircle, RefreshCw, AlertCircle, HardDrive, BarChart3 } from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function PerformanceStats() {
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(true);
  const [switching, setSwitching] = useState(false);
  const [error, setError] = useState(null);
  const [switchMessage, setSwitchMessage] = useState(null);

  const fetchHealth = async () => {
    try {
      const response = await fetch(`${API_URL}/health`);
      if (!response.ok) throw new Error("Health check returned non-200");
      const data = await response.json();
      setHealth(data);
      setError(null);
    } catch (err) {
      console.error(err);
      setError("Cannot connect to FastAPI server. Verify the server is running on port 8000.");
    } finally {
      setLoading(false);
    }
  };

  const changeModel = async (modelName) => {
    setSwitching(true);
    setSwitchMessage(null);
    try {
      const response = await fetch(`${API_URL}/change-model`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_name: modelName })
      });
      const data = await response.json();
      if (response.ok && data.success) {
        setSwitchMessage(`Successfully loaded ${modelName}`);
        setHealth(prev => prev ? { ...prev, active_model: modelName } : null);
      } else {
        throw new Error(data.detail || "Model switch failed");
      }
    } catch (err) {
      console.error(err);
      setError(`Failed to switch model to ${modelName}. Make sure the model weights are generated.`);
    } finally {
      setSwitching(false);
    }
  };

  useEffect(() => {
    fetchHealth();
    const interval = setInterval(fetchHealth, 10000); // refresh system health every 10s
    return () => clearInterval(interval);
  }, []);

  const modelsList = [
    { id: 'custom_cnn', name: 'Custom Fast CNN', desc: 'Optimized depthwise separable convs. Ideal for edge CPUs.', speed: '2.5 ms', accuracy: '95.8%' },
    { id: 'mobilenet_v2', name: 'MobileNetV2', desc: 'Pre-trained transfer model, balanced latency & depth.', speed: '5.2 ms', accuracy: '97.2%' },
    { id: 'efficientnet', name: 'EfficientNetB0', desc: 'Pre-trained compound scaled features. Highest accuracy.', speed: '9.4 ms', accuracy: '98.5%' },
    { id: 'rcnn', name: 'Recurrent CNN (RCNN)', desc: 'Liang & Hu style architecture using Recurrent Convolutional Layers (RCL) with weight sharing.', speed: '4.2 ms', accuracy: '93.3%' },
    { id: 'onnx', name: 'ONNX (Custom)', desc: 'ONNX runtime serialized. Enhanced CPU processing speeds.', speed: '1.2 ms', accuracy: '95.8%' },
    { id: 'onnx_quantized', name: 'Quantized INT8 ONNX', desc: 'INT8 compressed weights. Minimal memory footprint.', speed: '0.8 ms', accuracy: '95.3%' }
  ];

  return (
    <div className="space-y-6">
      
      {/* Hardware Health Monitors */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        
        {/* Model State */}
        <div className="glass-panel rounded-2xl p-5 flex items-start gap-4">
          <div className="p-3 bg-sky-950/50 rounded-xl border border-sky-800/30">
            <Cpu className="w-6 h-6 text-sky-400" />
          </div>
          <div className="flex-1 min-w-0">
            <span className="text-xs text-slate-500 font-semibold uppercase tracking-wider">Active Neural Model</span>
            {loading ? (
              <div className="h-6 w-32 bg-slate-900 animate-pulse rounded mt-2"></div>
            ) : (
              <h3 className="text-lg font-bold text-slate-100 mt-0.5 truncate uppercase">
                {health?.active_model || "None"}
              </h3>
            )}
            <span className="text-[10px] text-slate-400 font-medium bg-slate-900 border border-slate-800 px-2 py-0.5 rounded mt-2 inline-block">
              Device: {health?.device || "N/A"}
            </span>
          </div>
        </div>

        {/* GPU VRAM Allocator */}
        <div className="glass-panel rounded-2xl p-5 flex items-start gap-4">
          <div className="p-3 bg-violet-950/50 rounded-xl border border-violet-800/30">
            <HardDrive className="w-6 h-6 text-violet-400" />
          </div>
          <div className="flex-1 min-w-0">
            <span className="text-xs text-slate-500 font-semibold uppercase tracking-wider">GPU Diagnostics</span>
            {loading ? (
              <div className="h-6 w-32 bg-slate-900 animate-pulse rounded mt-2"></div>
            ) : (
              <h3 className="text-lg font-bold text-slate-100 mt-0.5 truncate">
                {health?.cuda_available ? health.gpu_name : "CPU ONLY"}
              </h3>
            )}
            <span className="text-[10px] text-slate-400 font-medium bg-slate-900 border border-slate-800 px-2 py-0.5 rounded mt-2 inline-block">
              VRAM Cache: {health?.vram_allocated_mb.toFixed(1)} MB
            </span>
          </div>
        </div>

        {/* Server Memory Allocator */}
        <div className="glass-panel rounded-2xl p-5 flex items-start gap-4">
          <div className="p-3 bg-emerald-950/50 rounded-xl border border-emerald-800/30">
            <BarChart3 className="w-6 h-6 text-emerald-400" />
          </div>
          <div className="flex-1 min-w-0">
            <span className="text-xs text-slate-500 font-semibold uppercase tracking-wider">System Host Load</span>
            {loading ? (
              <div className="h-6 w-32 bg-slate-900 animate-pulse rounded mt-2"></div>
            ) : (
              <h3 className="text-lg font-bold text-slate-100 mt-0.5">
                RAM Load: {health?.memory_usage_percent.toFixed(0)}%
              </h3>
            )}
            <span className="text-[10px] text-slate-400 font-medium bg-slate-900 border border-slate-800 px-2 py-0.5 rounded mt-2 inline-block">
              CPU Logical Cores: {health?.cpu_cores || 0}
            </span>
          </div>
        </div>
      </div>

      {/* Model Hot-swapping Panel */}
      <div className="glass-panel rounded-2xl p-6 glow-box-blue">
        <div className="flex justify-between items-center mb-6">
          <div>
            <h2 className="text-xl font-bold tracking-tight">Active Model Architecture</h2>
            <p className="text-slate-400 text-xs mt-1">
              Dynamically hot-swap active model instances running in backend memory. Choose optimized ONNX models for fast edge-inference.
            </p>
          </div>
          <button 
            onClick={fetchHealth} 
            disabled={loading}
            className="p-2 text-slate-400 hover:text-sky-400 rounded-lg hover:bg-slate-900 border border-slate-800 hover:border-slate-700 transition-all"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>

        {error && (
          <div className="mb-6 bg-red-950/20 border border-red-800/30 text-red-400 p-4 rounded-xl flex items-center gap-3">
            <AlertCircle className="w-5 h-5 text-red-500 shrink-0" />
            <span className="text-sm">{error}</span>
          </div>
        )}

        {switchMessage && (
          <div className="mb-6 bg-emerald-950/20 border border-emerald-800/30 text-emerald-400 p-4 rounded-xl flex items-center gap-3">
            <CheckCircle className="w-5 h-5 text-emerald-500 shrink-0" />
            <span className="text-sm font-medium">{switchMessage}</span>
          </div>
        )}

        <div className="grid grid-cols-1 gap-4">
          {modelsList.map((model) => {
            const isActive = health?.active_model === model.id;
            return (
              <div 
                key={model.id}
                className={`p-4 rounded-xl border transition-all flex flex-col md:flex-row md:items-center justify-between gap-4 ${
                  isActive 
                    ? 'bg-sky-950/10 border-sky-500 shadow-[0_0_15px_rgba(14,165,233,0.1)]' 
                    : 'bg-slate-900/20 border-slate-800/60 hover:border-slate-700'
                }`}
              >
                <div>
                  <div className="flex items-center gap-2">
                    <h3 className="font-bold text-slate-100">{model.name}</h3>
                    {isActive && (
                      <span className="bg-sky-500 text-black text-[9px] font-bold px-1.5 py-0.5 rounded tracking-wide uppercase">
                        Active
                      </span>
                    )}
                  </div>
                  <p className="text-slate-400 text-xs mt-1">{model.desc}</p>
                </div>

                <div className="flex items-center gap-4 shrink-0">
                  <div className="text-right text-xs">
                    <div className="text-slate-500">Benchmark Speed: <span className="font-mono font-semibold text-slate-300">{model.speed}</span></div>
                    <div className="text-slate-500">Test Accuracy: <span className="font-mono font-semibold text-emerald-400">{model.accuracy}</span></div>
                  </div>

                  <button
                    onClick={() => changeModel(model.id)}
                    disabled={isActive || switching}
                    className={`py-2 px-4 rounded-lg font-bold text-xs transition-all uppercase tracking-wide ${
                      isActive
                        ? 'bg-sky-950/50 text-sky-500 border border-sky-500/20 cursor-default'
                        : 'bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-800'
                    }`}
                  >
                    {isActive ? "Loaded" : "Activate"}
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
