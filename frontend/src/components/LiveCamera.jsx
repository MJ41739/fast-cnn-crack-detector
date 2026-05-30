import React, { useState, useRef, useEffect } from 'react';
import { Camera, VideoOff, RefreshCw, AlertTriangle, ShieldCheck, Zap } from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || "/api";

export default function LiveCamera() {
  const [active, setActive] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [fps, setFps] = useState(0);
  const [latency, setLatency] = useState(0);
  
  // Custom scanner controls
  const [threshold, setThreshold] = useState(0.5);
  const [smoothingFrames, setSmoothingFrames] = useState(5);
  const [useSquareCrop, setUseSquareCrop] = useState(true);
  const [resolution, setResolution] = useState("720p");
  const [scanAreaScale, setScanAreaScale] = useState(0.7); // 70% of viewport by default
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const animationFrameId = useRef(null);
  const lastFrameTime = useRef(performance.now());
  const processingRef = useRef(false);
  const isMounted = useRef(true);
  const historyRef = useRef([]);

  const getResolutionConstraints = (res) => {
    switch (res) {
      case "1080p":
        return { width: { ideal: 1920 }, height: { ideal: 1080 } };
      case "720p":
        return { width: { ideal: 1280 }, height: { ideal: 720 } };
      case "480p":
      default:
        return { width: { ideal: 640 }, height: { ideal: 480 } };
    }
  };

  // Start webcam stream
  const startCamera = async (currentRes = resolution) => {
    setError(null);
    setLoading(true);
    try {
      let stream;
      try {
        const resConstraints = getResolutionConstraints(currentRes);
        const constraints = {
          video: { 
            ...resConstraints,
            facingMode: "environment" // Prefer rear camera on mobile devices
          }
        };
        stream = await navigator.mediaDevices.getUserMedia(constraints);
      } catch (firstErr) {
        console.warn("Failed to get camera with optimal constraints, trying generic fallback:", firstErr);
        // Fall back to basic video request
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
      }

      if (!isMounted.current) {
        // Prevent stream leak if unmounted while getUserMedia was resolving
        stream.getTracks().forEach(track => track.stop());
        return;
      }

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setActive(true);
    } catch (err) {
      if (isMounted.current) {
        console.error(err);
        setError(`Failed to access camera (${err.name}: ${err.message}). Please ensure camera permissions are allowed and that the device is not in use by another application.`);
        setActive(false);
      }
    } finally {
      if (isMounted.current) {
        setLoading(false);
      }
    }
  };

  // Stop webcam stream
  const stopCamera = () => {
    setActive(false);
    setPrediction(null);
    setFps(0);
    setLatency(0);
    historyRef.current = []; // Clear smoothing history
    
    if (animationFrameId.current) {
      cancelAnimationFrame(animationFrameId.current);
    }
    
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  };

  // Toggle state
  const handleToggle = () => {
    if (active) {
      stopCamera();
    } else {
      startCamera();
    }
  };

  const handleResolutionChange = async (newRes) => {
    setResolution(newRes);
    if (active) {
      stopCamera();
      // Allow a brief moment for the camera device to release
      setTimeout(() => {
        startCamera(newRes);
      }, 300);
    }
  };

  // Clean up on unmount
  useEffect(() => {
    isMounted.current = true;
    return () => {
      isMounted.current = false;
      stopCamera();
    };
  }, []);

  // Frame processing loop
  useEffect(() => {
    if (!active) return;

    const processFrame = async () => {
      if (!active || !videoRef.current || !canvasRef.current) return;
      
      const video = videoRef.current;
      const canvas = canvasRef.current;
      
      // Ensure video is playing and has valid dimensions
      if (video.readyState === video.HAVE_ENOUGH_DATA && !processingRef.current) {
        processingRef.current = true;
        
        const ctx = canvas.getContext('2d');
        
        if (useSquareCrop) {
          // Crop centered square with custom scale (digital zoom)
          const baseSize = Math.min(video.videoWidth, video.videoHeight);
          const size = Math.round(baseSize * scanAreaScale);
          
          canvas.width = size;
          canvas.height = size;
          
          const sx = (video.videoWidth - size) / 2;
          const sy = (video.videoHeight - size) / 2;
          ctx.drawImage(video, sx, sy, size, size, 0, 0, size, size);
        } else {
          // Standard full-frame (potentially squished in backend Resize)
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        }
        
        // Capture frame as blob
        canvas.toBlob(async (blob) => {
          if (!blob) {
            processingRef.current = false;
            return;
          }
          
          const formData = new FormData();
          formData.append("file", blob, "frame.jpg");
          
          try {
            const startPredictTime = performance.now();
            
            // Call prediction endpoint (disable Grad-CAM for speed)
            const response = await fetch(`${API_URL}/predict?heatmap=false`, {
              method: "POST",
              body: formData,
            });
            
            if (response.ok) {
              const data = await response.json();
              
              // Apply temporal smoothing if enabled
              let finalProb = data.probability;
              if (smoothingFrames > 1) {
                historyRef.current.push(data.probability);
                if (historyRef.current.length > smoothingFrames) {
                  historyRef.current.shift();
                }
                finalProb = historyRef.current.reduce((a, b) => a + b, 0) / historyRef.current.length;
              } else {
                historyRef.current = [];
              }
              
              setPrediction({
                ...data,
                probability: finalProb
              });
              
              // Calculate performance metrics
              const endPredictTime = performance.now();
              const currentLatency = endPredictTime - startPredictTime;
              setLatency(currentLatency);
              
              const currentFrameTime = performance.now();
              const delta = currentFrameTime - lastFrameTime.current;
              const currentFps = 1000.0 / delta;
              setFps(Math.round(currentFps));
              lastFrameTime.current = currentFrameTime;
            }
          } catch (err) {
            console.error("Frame inference error:", err);
          } finally {
            processingRef.current = false;
          }
        }, "image/jpeg", 0.85); // slightly higher quality for better accuracy
      }
      
      // Schedule next frame
      // Introduce a slight delay (e.g. 50ms) to target 15-20 FPS and avoid overloading network
      setTimeout(() => {
        if (active) {
          animationFrameId.current = requestAnimationFrame(processFrame);
        }
      }, 60);
    };

    animationFrameId.current = requestAnimationFrame(processFrame);

    return () => {
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current);
      }
    };
  }, [active, useSquareCrop, smoothingFrames, scanAreaScale]); // add dependencies

  const isCrackFound = prediction && prediction.probability >= threshold;
  const predictionText = prediction 
    ? (prediction.probability >= threshold ? "Crack Detected" : "No Crack Detected") 
    : "";
  const confidenceValue = prediction
    ? (isCrackFound ? prediction.probability : 1.0 - prediction.probability)
    : 0;

  return (
    <div className="space-y-6">
      <div className="glass-panel rounded-2xl p-6 glow-box-blue flex flex-col md:flex-row gap-6">
        
        {/* Stream Viewport */}
        <div className="flex-1 flex flex-col items-center">
          <div className="relative w-full aspect-video rounded-xl bg-slate-900 border border-slate-800 overflow-hidden flex items-center justify-center">
            
            {/* Native Video Feed */}
            <video 
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className={`w-full h-full object-cover ${active ? 'block' : 'hidden'}`}
            />
            
            {/* Offscreen frame grabber canvas (kept hidden) */}
            <canvas ref={canvasRef} className="hidden" />

            {/* Center square scanning target guide with dynamic sizing */}
            {active && useSquareCrop && (
              <div 
                className="absolute border-2 border-dashed border-sky-500/40 rounded-2xl pointer-events-none flex items-center justify-center animate-pulse"
                style={{
                  width: `${scanAreaScale * 80}%`,
                  maxWidth: `${scanAreaScale * 400}px`,
                  aspectRatio: "1/1"
                }}
              >
                <div className="absolute w-4 h-4 border-t-2 border-l-2 border-sky-400 top-0 left-0"></div>
                <div className="absolute w-4 h-4 border-t-2 border-r-2 border-sky-400 top-0 right-0"></div>
                <div className="absolute w-4 h-4 border-b-2 border-l-2 border-sky-400 bottom-0 left-0"></div>
                <div className="absolute w-4 h-4 border-b-2 border-r-2 border-sky-400 bottom-0 right-0"></div>
                <div className="w-1.5 h-1.5 bg-sky-400 rounded-full opacity-60"></div>
              </div>
            )}

            {/* Inactive State Screen */}
            {!active && !loading && (
              <div className="text-center p-8 text-slate-500">
                <VideoOff className="w-14 h-14 mx-auto mb-4 text-slate-700" />
                <p className="font-semibold text-slate-400">Camera stream offline</p>
                <p className="text-xs text-slate-500 mt-1">Start camera feed to begin scan</p>
              </div>
            )}

            {loading && (
              <div className="text-center">
                <RefreshCw className="w-10 h-10 mx-auto text-sky-500 animate-spin mb-4" />
                <p className="font-semibold text-slate-400">Requesting media streams...</p>
              </div>
            )}

            {/* Glowing Border Overlay indicating status */}
            {active && prediction && (
              <div className={`absolute inset-0 pointer-events-none transition-all duration-300 border-4 ${
                isCrackFound 
                  ? 'border-red-500/80 shadow-[inset_0_0_40px_rgba(239,68,68,0.5)] animate-pulse' 
                  : 'border-emerald-500/50 shadow-[inset_0_0_20px_rgba(16,185,129,0.2)]'
              }`} />
            )}

            {/* Real-time Diagnostics overlay tags */}
            {active && (
              <div className="absolute top-3 left-3 flex gap-2 pointer-events-none">
                <div className="bg-black/75 backdrop-blur-md px-2.5 py-1.5 rounded-lg text-xs font-semibold text-slate-300 border border-white/5">
                  FPS: <span className="text-sky-400 font-mono">{fps}</span>
                </div>
                <div className="bg-black/75 backdrop-blur-md px-2.5 py-1.5 rounded-lg text-xs font-semibold text-slate-300 border border-white/5">
                  Latency: <span className="text-amber-500 font-mono">{latency.toFixed(0)}ms</span>
                </div>
              </div>
            )}

            {active && prediction && (
              <div className="absolute bottom-3 left-3 pointer-events-none">
                <div className={`backdrop-blur-md px-3.5 py-2 rounded-lg text-xs font-bold uppercase tracking-wider border flex items-center gap-1.5 ${
                  isCrackFound 
                    ? 'bg-red-950/80 border-red-800/40 text-red-200' 
                    : 'bg-emerald-950/80 border-emerald-800/40 text-emerald-200'
                }`}>
                  {isCrackFound ? <AlertTriangle className="w-4 h-4 text-red-500 animate-bounce" /> : <ShieldCheck className="w-4 h-4 text-emerald-500" />}
                  {predictionText} ({(confidenceValue * 100).toFixed(0)}%)
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Info & Settings Panel */}
        <div className="w-full md:w-[320px] flex flex-col justify-between text-slate-200">
          <div className="space-y-6">
            <div>
              <h2 className="text-xl font-bold tracking-tight">Real-time Scanner</h2>
              <p className="text-slate-400 text-xs mt-1 leading-relaxed">
                Connect your device's camera for active site inspection. Move the lens over concrete walls or floors; cracks will automatically be flagged.
              </p>
            </div>

            {/* Tuning Controls */}
            {active && (
              <div className="space-y-4 pt-4 border-t border-slate-800">
                <h3 className="text-xs font-bold uppercase text-slate-500 tracking-wider">Scanner Tuning</h3>
                
                {/* Resolution Selector */}
                <div className="space-y-1.5">
                  <span className="text-xs font-semibold text-slate-300 block">Camera Resolution</span>
                  <div className="grid grid-cols-3 gap-2">
                    {["480p", "720p", "1080p"].map((res) => (
                      <button
                        key={res}
                        onClick={() => handleResolutionChange(res)}
                        className={`py-1.5 rounded-lg text-xs font-semibold border transition-all ${
                          resolution === res
                            ? "bg-sky-600 border-sky-500 text-white font-bold"
                            : "bg-slate-900 border-slate-800 text-slate-400 hover:bg-slate-850"
                        }`}
                      >
                        {res}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Scan Area Size (Digital Zoom) */}
                {useSquareCrop && (
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-xs font-semibold text-slate-300">
                      <span>Digital Scan Zoom</span>
                      <span className="text-sky-400 font-mono">{Math.round((1 - scanAreaScale) * 100)}%</span>
                    </div>
                    <input
                      type="range"
                      min="0.3"
                      max="1.0"
                      step="0.05"
                      value={scanAreaScale}
                      onChange={(e) => setScanAreaScale(parseFloat(e.target.value))}
                      className="w-full h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-sky-500"
                    />
                    <p className="text-[10px] text-slate-500 leading-relaxed">
                      Shrink the reticle size to crop closer around cracks. Preserves fine details (acts like a macro lens).
                    </p>
                  </div>
                )}

                {/* Sensitivity Slider */}
                <div className="space-y-1.5">
                  <div className="flex justify-between text-xs font-semibold text-slate-300">
                    <span>Sensitivity (Threshold)</span>
                    <span className="text-sky-400 font-mono">{(threshold * 100).toFixed(0)}%</span>
                  </div>
                  <input
                    type="range"
                    min="0.10"
                    max="0.90"
                    step="0.05"
                    value={threshold}
                    onChange={(e) => setThreshold(parseFloat(e.target.value))}
                    className="w-full h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-sky-500"
                  />
                  <p className="text-[10px] text-slate-500 leading-relaxed">
                    Lower values catch more hairline cracks (higher recall). Higher values reduce false alerts on shadows (higher precision).
                  </p>
                </div>

                {/* Smoothing Slider */}
                <div className="space-y-1.5">
                  <div className="flex justify-between text-xs font-semibold text-slate-300">
                    <span>Temporal Smoothing</span>
                    <span className="text-sky-400 font-mono">
                      {smoothingFrames === 1 ? "Disabled" : `${smoothingFrames} Frames`}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="15"
                    step="1"
                    value={smoothingFrames}
                    onChange={(e) => setSmoothingFrames(parseInt(e.target.value))}
                    className="w-full h-1 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-sky-500"
                  />
                  <p className="text-[10px] text-slate-500 leading-relaxed">
                    Averages predictions over consecutive frames to filter out flickering and noise.
                  </p>
                </div>

                {/* Square Crop Toggle */}
                <div className="flex items-center gap-2 py-1">
                  <input
                    type="checkbox"
                    id="square_crop"
                    checked={useSquareCrop}
                    onChange={(e) => setUseSquareCrop(e.target.checked)}
                    className="w-4 h-4 rounded text-sky-500 bg-slate-950 border-slate-800 focus:ring-sky-500 cursor-pointer"
                  />
                  <label htmlFor="square_crop" className="cursor-pointer text-xs font-semibold text-slate-300 select-none">
                    Enable Square Center Crop
                  </label>
                </div>
                <p className="text-[10px] text-slate-500 leading-relaxed">
                  Crops a 1:1 aspect ratio to match the neural network's training format. Dramatically improves classification accuracy.
                </p>
              </div>
            )}

            {!active && (
              <div className="space-y-4">
                <h3 className="text-xs font-bold uppercase text-slate-500 tracking-wider">Calibration Instructions</h3>
                <ul className="text-xs text-slate-400 space-y-2 list-disc list-inside">
                  <li>Ensure sufficient environmental brightness.</li>
                  <li>Hold camera perpendicular to inspect surfaces.</li>
                  <li>Keep camera ~1 foot away for optimal scale.</li>
                  <li>Slow movements prevent motion blur distortions.</li>
                </ul>
              </div>
            )}

            {error && (
              <div className="bg-red-950/20 border border-red-800/30 text-red-400 p-3 rounded-xl text-xs flex gap-2">
                <AlertTriangle className="w-4 h-4 text-red-500 shrink-0" />
                <span>{error}</span>
              </div>
            )}
          </div>

          <div className="mt-8">
            <button
              onClick={handleToggle}
              disabled={loading}
              className={`w-full py-3 px-4 font-bold text-white rounded-xl shadow-lg transition-all flex items-center justify-center gap-2 ${
                active 
                  ? 'bg-red-600 hover:bg-red-500 shadow-red-900/10' 
                  : 'bg-sky-600 hover:bg-sky-500 shadow-sky-950/10'
              }`}
            >
              <Camera className="w-5 h-5" />
              {active ? "Stop Scanner Feed" : "Start Live Scanner"}
            </button>
          </div>
        </div>
        
      </div>
    </div>
  );
}
