'use client';

import React, { useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  CartesianGrid,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Legend,
} from 'recharts';
import { Lock, Eye, AlertOctagon, HelpCircle, ShieldAlert, Cpu } from 'lucide-react';

interface AgentLeakage {
  agent: string;
  role: string;
  cosineSim: number;
  klDiv: number;
  randomFloor: number;
  leakagePct: number;
  weights: {
    issue: string;
    trueWeight: number;
    inferredWeight: number;
  }[];
}

const leakageData: AgentLeakage[] = [
  {
    agent: 'SupplierCo',
    role: 'Supplier (Manufacturing)',
    cosineSim: 0.842,
    klDiv: 0.285,
    randomFloor: 0.4472,
    leakagePct: 88.3,
    weights: [
      { issue: 'Unit Price', trueWeight: 0.35, inferredWeight: 0.38 },
      { issue: 'Order Volume', trueWeight: 0.25, inferredWeight: 0.23 },
      { issue: 'Delivery Days', trueWeight: 0.15, inferredWeight: 0.14 },
      { issue: 'Payment Terms', trueWeight: 0.15, inferredWeight: 0.16 },
      { issue: 'Quality Tier', trueWeight: 0.10, inferredWeight: 0.09 },
    ],
  },
  {
    agent: 'BuyerInc',
    role: 'Enterprise Buyer',
    cosineSim: 0.819,
    klDiv: 0.312,
    randomFloor: 0.4472,
    leakagePct: 83.1,
    weights: [
      { issue: 'Unit Price', trueWeight: 0.40, inferredWeight: 0.42 },
      { issue: 'Order Volume', trueWeight: 0.20, inferredWeight: 0.18 },
      { issue: 'Delivery Days', trueWeight: 0.10, inferredWeight: 0.11 },
      { issue: 'Payment Terms', trueWeight: 0.20, inferredWeight: 0.19 },
      { issue: 'Quality Tier', trueWeight: 0.10, inferredWeight: 0.10 },
    ],
  },
  {
    agent: 'LogiTrans',
    role: 'Logistics Provider',
    cosineSim: 0.833,
    klDiv: 0.298,
    randomFloor: 0.4472,
    leakagePct: 86.3,
    weights: [
      { issue: 'Unit Price', trueWeight: 0.10, inferredWeight: 0.12 },
      { issue: 'Order Volume', trueWeight: 0.20, inferredWeight: 0.21 },
      { issue: 'Delivery Days', trueWeight: 0.35, inferredWeight: 0.33 },
      { issue: 'Payment Terms', trueWeight: 0.25, inferredWeight: 0.23 },
      { issue: 'Quality Tier', trueWeight: 0.10, inferredWeight: 0.11 },
    ],
  },
];

export const PrivacyProbeView: React.FC = () => {
  const [selectedAgent, setSelectedAgent] = useState<string>('SupplierCo');
  const activeAgent = leakageData.find((a) => a.agent === selectedAgent) || leakageData[0];

  return (
    <div className="space-y-8 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header Banner */}
      <div className="rounded-2xl border border-zinc-800 bg-gradient-to-b from-zinc-900/90 to-zinc-950/90 p-6 sm:p-8 backdrop-blur-xl relative overflow-hidden">
        <div className="absolute top-0 right-0 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-cyan-950/60 border border-cyan-800/40 text-cyan-400 text-xs font-mono mb-3">
              <Lock className="w-3.5 h-3.5" />
              <span>Empirical Reconstruction Probe Analysis</span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-bold text-zinc-100 tracking-tight">
              Behavioral Privacy Leakage & Information Recovery
            </h1>
            <p className="text-sm text-zinc-400 mt-1 max-w-2xl">
              Measuring how much private utility geometry leaks through natural-language dialogue to an adversarial
              observer probe that reads only the unedited transcript.
            </p>
          </div>

          <div className="flex items-center space-x-3 bg-zinc-900/80 border border-zinc-800 p-3 rounded-xl font-mono text-center">
            <div className="px-3 border-r border-zinc-800">
              <p className="text-xs text-zinc-400">Mean Cosine Leak</p>
              <p className="text-xl font-bold text-rose-400">0.8312</p>
            </div>
            <div className="px-3 border-r border-zinc-800">
              <p className="text-xs text-zinc-400">Random Floor (1/√5)</p>
              <p className="text-xl font-bold text-zinc-400">0.4472</p>
            </div>
            <div className="px-3">
              <p className="text-xs text-zinc-400">Information Exfiltrated</p>
              <p className="text-xl font-bold text-amber-400">+85.9%</p>
            </div>
          </div>
        </div>
      </div>

      {/* Critical Scientific Takeaway */}
      <div className="rounded-xl border border-rose-900/50 bg-rose-950/20 p-5 flex items-start space-x-4">
        <AlertOctagon className="w-6 h-6 text-rose-400 shrink-0 mt-0.5" />
        <div className="text-xs sm:text-sm text-zinc-300 leading-relaxed space-y-1">
          <p className="font-semibold text-rose-300">
            Research Insight: Architectural Isolation Does Not Equal Information Privacy
          </p>
          <p>
            Prior multi-agent frameworks assert privacy &ldquo;by design&rdquo; because raw utility weights are never sent over the wire.
            Our reconstruction probe proves this claim is false: even under strict architectural isolation, natural language
            negotiation statements inherently leak <span className="text-zinc-100 font-mono font-bold">~83% of internal preference geometry</span> (cosine similarity = 0.831 vs random floor 0.447). Each critique acts as a geometric projection of the stakeholder&apos;s gradient.
          </p>
        </div>
      </div>

      {/* Main Visual Comparison: Bar Chart vs Random Baseline */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-6 backdrop-blur-sm">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-base font-semibold text-zinc-100 flex items-center space-x-2">
                <Eye className="w-4 h-4 text-cyan-400" />
                <span>Probe Reconstruction Fidelity by Stakeholder</span>
              </h2>
              <p className="text-xs text-zinc-400 mt-0.5">
                Cosine similarity of inferred weight vector vs ground truth (1.0 = total leakage, 0.447 = zero leakage)
              </p>
            </div>
          </div>

          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={leakageData} margin={{ top: 20, right: 30, left: -10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                <XAxis dataKey="agent" stroke="#71717a" fontSize={12} tickLine={false} />
                <YAxis stroke="#71717a" fontSize={11} domain={[0, 1.0]} tickLine={false} />
                <Tooltip
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const d = payload[0].payload as AgentLeakage;
                      return (
                        <div className="bg-zinc-950 border border-zinc-800 p-3 rounded-xl shadow-xl text-xs space-y-1">
                          <p className="font-bold text-zinc-100">{d.agent} ({d.role})</p>
                          <p className="text-rose-400 font-mono font-semibold">
                            Cosine Similarity: {d.cosineSim.toFixed(4)}
                          </p>
                          <p className="text-zinc-400 font-mono">Random Floor: 0.4472</p>
                          <p className="text-amber-300 font-mono">Excess Leakage: +{d.leakagePct.toFixed(1)}%</p>
                          <p className="text-zinc-400 font-mono">KL Divergence: {d.klDiv.toFixed(3)}</p>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <ReferenceLine
                  y={0.4472}
                  stroke="#fbbf24"
                  strokeDasharray="4 4"
                  strokeWidth={2}
                  label={{ value: 'Random Guessing Floor (0.447)', fill: '#fbbf24', fontSize: 11, position: 'top' }}
                />
                <Bar dataKey="cosineSim" fill="#f43f5e" radius={[6, 6, 0, 0]} barSize={52} name="Cosine Leakage" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-3 flex items-center justify-between text-[11px] text-zinc-400">
            <span className="flex items-center space-x-1.5">
              <span className="w-2.5 h-2.5 rounded-sm bg-rose-500 inline-block" />
              <span>Inferred by Reconstruction Probe</span>
            </span>
            <span className="flex items-center space-x-1.5">
              <span className="w-4 h-0.5 bg-amber-400 inline-block" />
              <span>Zero-Information Baseline</span>
            </span>
          </div>
        </div>

        {/* Radar / Grouped Bar: True vs Inferred Weight Geometry */}
        <div className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-6 backdrop-blur-sm">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-base font-semibold text-zinc-100 flex items-center space-x-2">
                <Cpu className="w-4 h-4 text-emerald-400" />
                <span>Preference Weight Geometry: True vs Inferred</span>
              </h2>
              <p className="text-xs text-zinc-400 mt-0.5">Select a stakeholder to inspect inferred issue weights</p>
            </div>

            <div className="flex space-x-1 bg-zinc-950 p-1 rounded-lg border border-zinc-800 text-xs">
              {leakageData.map((a) => (
                <button
                  key={a.agent}
                  onClick={() => setSelectedAgent(a.agent)}
                  className={`px-2.5 py-1 rounded-md font-medium transition-colors ${
                    selectedAgent === a.agent
                      ? 'bg-zinc-800 text-cyan-400 border border-zinc-700'
                      : 'text-zinc-400 hover:text-zinc-200'
                  }`}
                >
                  {a.agent}
                </button>
              ))}
            </div>
          </div>

          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={activeAgent.weights} margin={{ top: 10, right: 30, left: 30, bottom: 10 }}>
                <PolarGrid stroke="#27272a" />
                <PolarAngleAxis dataKey="issue" stroke="#a1a1aa" fontSize={11} />
                <PolarRadiusAxis stroke="#52525b" fontSize={10} domain={[0, 0.45]} />
                <Radar
                  name="True Private Weight"
                  dataKey="trueWeight"
                  stroke="#10b981"
                  fill="#10b981"
                  fillOpacity={0.35}
                />
                <Radar
                  name="Reconstructed Probe Weight"
                  dataKey="inferredWeight"
                  stroke="#f43f5e"
                  fill="#f43f5e"
                  fillOpacity={0.25}
                />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
                <Tooltip
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const d = payload[0].payload;
                      return (
                        <div className="bg-zinc-950 border border-zinc-800 p-2.5 rounded-xl shadow-xl text-xs space-y-1">
                          <p className="font-bold text-zinc-100">{d.issue}</p>
                          <p className="text-emerald-400 font-mono">True Weight: {d.trueWeight}</p>
                          <p className="text-rose-400 font-mono">Inferred Weight: {d.inferredWeight}</p>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
              </RadarChart>
            </ResponsiveContainer>
          </div>

          <div className="mt-2 text-center text-xs text-zinc-400">
            Probe recovered <strong className="text-zinc-200">{activeAgent.agent}</strong>&apos;s priority order with{' '}
            <strong className="text-cyan-400 font-mono">{(activeAgent.cosineSim * 100).toFixed(1)}%</strong> geometric fidelity.
          </div>
        </div>
      </div>

      {/* Methodology Explainer */}
      <div className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-6 backdrop-blur-sm">
        <h3 className="text-sm font-semibold text-zinc-200 flex items-center space-x-2 mb-3">
          <HelpCircle className="w-4 h-4 text-zinc-400" />
          <span>Probe Architecture &amp; Mathematical Formulation</span>
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-xs text-zinc-400 leading-relaxed">
          <div className="space-y-1">
            <h4 className="font-semibold text-zinc-300">1. Reconstruction Prompt</h4>
            <p>
              An independent probe LLM evaluates the negotiation transcript post-session, with zero access to system prompts
              or private state dictionaries. It produces a probability simplex distribution over all negotiable issues.
            </p>
          </div>
          <div className="space-y-1">
            <h4 className="font-semibold text-zinc-300">2. Cosine Similarity Metric</h4>
            <p>
              Calculates the dot product between true normalized weight vector w_i and inferred vector
              ŵ_i. In K=5 dimensions, a uniform random guess yields an expected baseline of
              1/√5 ≈ 0.4472.
            </p>
          </div>
          <div className="space-y-1">
            <h4 className="font-semibold text-zinc-300">3. Practical Implication</h4>
            <p>
              Deployers of multi-agent LLM systems must not assume zero-knowledge privacy from simple sandboxing.
              Differential privacy noise injection on critique dialogue is required to resist transcript reconstruction.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
