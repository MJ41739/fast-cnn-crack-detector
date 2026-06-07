import React, { useState } from 'react';
import { Camera, Layers, Play, Settings, Shield, Activity } from 'lucide-react';
import DragDropUpload from './components/DragDropUpload';
import BatchUpload from './components/BatchUpload';
import LiveCamera from './components/LiveCamera';
import PerformanceStats from './components/PerformanceStats';

export default function App() {
  const [activeTab, setActiveTab] = useState('single');

  const tabs = [
    { id: 'single', name: 'Single Scan', icon: Shield, component: DragDropUpload },
    { id: 'batch', name: 'Batch Inspector', icon: Layers, component: BatchUpload },
    { id: 'live', name: 'Live Stream', icon: Camera, component: LiveCamera },
    { id: 'perf', name: 'Performance & Config', icon: Activity, component: PerformanceStats }
  ];

  return (
    <div className="min-h-screen flex flex-col justify-between p-4 sm:p-6 lg:p-8 max-w-7xl mx-auto space-y-8">
      {/* Header Panel */}
      <header className="glass-panel rounded-2xl p-6 glow-box-blue flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <div className="p-3 bg-gradient-to-tr from-sky-500 to-indigo-500 rounded-xl shadow-lg shadow-sky-500/20">
            <Shield className="w-8 h-8 text-black fill-black" />
          </div>
          <div>
            <h1 className="text-2xl font-extrabold tracking-tight uppercase bg-gradient-to-r from-white via-slate-100 to-sky-400 bg-clip-text text-transparent">
              Crack Detection System
            </h1>
            <p className="text-slate-400 text-xs mt-0.5 font-medium tracking-wide">
              High-Accuracy Fast CNN Edge Defect Inspector
            </p>
          </div>
        </div>

        {/* Tab Selector Links */}
        <nav className="flex bg-slate-950/80 border border-slate-900 rounded-xl p-1 shrink-0 w-full sm:w-auto overflow-x-auto">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex-1 sm:flex-none flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-xs font-bold transition-all whitespace-nowrap ${isActive
                  ? 'bg-sky-600 text-white shadow-md'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/40'
                  }`}
              >
                <Icon className="w-4 h-4" />
                {tab.name}
              </button>
            );
          })}
        </nav>
      </header>

      {/* Main Dashboard Workspace */}
      <main className="flex-1">
        {tabs.map((tab) => {
          if (tab.id !== activeTab) return null;
          const Component = tab.component;
          return (
            <div key={tab.id} className="transition-all duration-300 ease-in-out">
              <Component />
            </div>
          );
        })}
      </main>

      {/* Footer Metrics */}
      <footer className="text-center py-4 text-[10px] text-slate-600 font-semibold tracking-wider uppercase flex flex-col sm:flex-row items-center justify-between gap-2 border-t border-slate-900/50 pt-6">
        <span>Industrial Infrastructure Monitoring System</span>
        {/* <span>© 2026 INDUSTRIAL DEFECT SYSTEMS INC.</span> */}
        {/* <span> */}
        {/* MLOps Architect: PyTorch CUDA AMP & ONNX Runtime Edge optimized */}
        {/* </span> */}
      </footer>
    </div>
  );
}
