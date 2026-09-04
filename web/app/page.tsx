'use client';

import React, { useState, useEffect } from 'react';
import { Navbar } from '@/components/Navbar';
import { NegotiationRunner } from '@/components/NegotiationRunner';
import { ResultsDashboard } from '@/components/ResultsDashboard';
import { PrivacyProbeView } from '@/components/PrivacyProbeView';
import { TrialExplorer } from '@/components/TrialExplorer';
import { ScenarioBuilder } from '@/components/ScenarioBuilder';
import { checkBackendHealth } from '@/lib/api';

export default function Home() {
  const [activeTab, setActiveTab] = useState<string>('benchmarks');
  const [backendOnline, setBackendOnline] = useState<boolean>(false);

  useEffect(() => {
    async function checkHealth() {
      const isUp = await checkBackendHealth();
      setBackendOnline(isUp);
    }
    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 selection:bg-emerald-500/30 selection:text-emerald-300 font-sans antialiased w-full max-w-full overflow-x-hidden">
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        backendOnline={backendOnline}
      />

      <main className="pb-16 w-full max-w-full overflow-x-hidden">
        {activeTab === 'runner' && <NegotiationRunner />}
        {activeTab === 'benchmarks' && <ResultsDashboard />}
        {activeTab === 'privacy' && <PrivacyProbeView />}
        {activeTab === 'logs' && <TrialExplorer />}
        {activeTab === 'builder' && <ScenarioBuilder />}
      </main>

      {/* Footer */}
      <footer className="border-t border-zinc-800/80 bg-zinc-950/60 py-8 text-center text-xs text-zinc-500">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-4">
          <p>
            Consensus Engine &middot; Autonomous Multi-Agent Negotiation Under Private Information
          </p>
          <div className="flex items-center space-x-4">
            <span className="text-zinc-600">Frontend on Vercel &middot; Backend on Render/Fly</span>
            <a
              href="https://github.com/Saksham-19-cyber/Consensus-Engine"
              target="_blank"
              rel="noreferrer"
              className="text-emerald-400 hover:underline"
            >
              GitHub Master
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}
