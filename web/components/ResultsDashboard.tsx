'use client';

import React, { useState, useEffect } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  ErrorBar,
  AreaChart,
  Area,
  Line,
} from 'recharts';
import { Award, AlertTriangle, ShieldCheck, Scale, Info } from 'lucide-react';
import { getEvalReport } from '@/lib/api';

interface BenchmarkMethod {
  name: string;
  label: string;
  pareto: number;
  paretoLow: number;
  paretoHigh: number;
  paretoError: [number, number]; // [down, up]
  nash: number;
  nashLow: number;
  nashHigh: number;
  nashError: [number, number];
  agreement: number;
  wilcoxonPareto: string;
  wilcoxonNash: string;
  infoLevel: string;
  rounds: number;
}

// Generated strictly from results/benchmark_business_deal.jsonl (N=30, seeds 42-71)
const defaultBenchmarkData: BenchmarkMethod[] = [
  {
    name: 'consensus_engine',
    label: 'Consensus Engine (Ours)',
    pareto: 0.9156,
    paretoLow: 0.9022,
    paretoHigh: 0.9274,
    paretoError: [0.0134, 0.0118],
    nash: 0.3990,
    nashLow: 0.3807,
    nashHigh: 0.4162,
    nashError: [0.0183, 0.0172],
    agreement: 100.0,
    wilcoxonPareto: '*(Engine)*',
    wilcoxonNash: '*(Engine)*',
    infoLevel: 'Private Utilities',
    rounds: 1.0,
  },
  {
    name: 'public_midpoint',
    label: 'Public Range Midpoint',
    pareto: 0.9082,
    paretoLow: 0.8973,
    paretoHigh: 0.9189,
    paretoError: [0.0109, 0.0107],
    nash: 0.3885,
    nashLow: 0.3752,
    nashHigh: 0.4011,
    nashError: [0.0133, 0.0126],
    agreement: 100.0,
    wilcoxonPareto: 'p = 0.0017 (**)',
    wilcoxonNash: 'p = 0.0015 (**)',
    infoLevel: 'Public Bounds Only',
    rounds: 0.0,
  },
  {
    name: 'naive_average',
    label: 'Private Ideal Average',
    pareto: 0.9525,
    paretoLow: 0.9427,
    paretoHigh: 0.9616,
    paretoError: [0.0098, 0.0091],
    nash: 0.4532,
    nashLow: 0.4357,
    nashHigh: 0.4697,
    nashError: [0.0175, 0.0165],
    agreement: 100.0,
    wilcoxonPareto: 'p < 1e-6 (**)',
    wilcoxonNash: 'p < 1e-6 (**)',
    infoLevel: 'Semi-Oracle (Private Ideals)',
    rounds: 0.0,
  },
  {
    name: 'nash_bargaining',
    label: 'Nash Bargaining Solution',
    pareto: 0.9991,
    paretoLow: 0.9984,
    paretoHigh: 0.9996,
    paretoError: [0.0007, 0.0005],
    nash: 0.5481,
    nashLow: 0.5370,
    nashHigh: 0.5583,
    nashError: [0.0111, 0.0102],
    agreement: 100.0,
    wilcoxonPareto: 'p < 1e-6 (**)',
    wilcoxonNash: 'p < 1e-6 (**)',
    infoLevel: 'Full Cooperative Oracle',
    rounds: 0.0,
  },
  {
    name: 'random_uniform',
    label: 'Random Uniform',
    pareto: 0.8205,
    paretoLow: 0.7960,
    paretoHigh: 0.8441,
    paretoError: [0.0245, 0.0236],
    nash: 0.2887,
    nashLow: 0.2624,
    nashHigh: 0.3150,
    nashError: [0.0263, 0.0263],
    agreement: 96.7,
    wilcoxonPareto: 'p < 1e-6 (**)',
    wilcoxonNash: 'p < 1e-6 (**)',
    infoLevel: 'Random Guess',
    rounds: 0.0,
  },
];

// Reservation data from results/reservation_sweep.jsonl and deterministic baselines
const reservationData = [
  { threshold: 'r = 0.40', rVal: 0.40, midpointBreach: 0.0, ceBreach: 0.0, midpointAgree: 100.0, ceAgree: 100.0, ceNote: '2/2 trials' },
  { threshold: 'r = 0.55', rVal: 0.55, midpointBreach: 4.0, ceBreach: 0.0, midpointAgree: 96.0, ceAgree: null, ceNote: 'Not yet run' },
  { threshold: 'r = 0.60', rVal: 0.60, midpointBreach: 33.0, ceBreach: 0.0, midpointAgree: 67.0, ceAgree: null, ceNote: 'Not yet run' },
  { threshold: 'r = 0.65', rVal: 0.65, midpointBreach: 78.0, ceBreach: 0.0, midpointAgree: 22.0, ceAgree: null, ceNote: 'Not yet run' },
  { threshold: 'r = 0.70', rVal: 0.70, midpointBreach: 100.0, ceBreach: 0.0, midpointAgree: 0.0, ceAgree: null, ceNote: 'Not yet run' },
];

// Deterministic baseline arithmetic, no LLM
const anchorData = [
  { level: '+10%', stated: 9.0, naiveShift: 3.0, naiveCapture: 3.3, midShift: 0.0 },
  { level: '+25%', stated: 22.5, naiveShift: 7.5, naiveCapture: 8.3, midShift: 0.0 },
  { level: '+50%', stated: 45.0, naiveShift: 15.0, naiveCapture: 16.7, midShift: 0.0 },
  { level: '+75%', stated: 67.5, naiveShift: 22.5, naiveCapture: 25.0, midShift: 0.0 },
];

export const ResultsDashboard: React.FC = () => {
  const [metricTab, setMetricTab] = useState<'pareto' | 'nash'>('pareto');
  const [data, setData] = useState<BenchmarkMethod[]>(defaultBenchmarkData);

  useEffect(() => {
    async function fetchReport() {
      try {
        const report = await getEvalReport('business_deal');
        if (report && report.summary && Object.keys(report.summary).length > 0) {
          const s = report.summary;
          const methods = Object.keys(s);
          const mapped: BenchmarkMethod[] = methods.map((m) => {
            const row = s[m];
            const pMean = row.mean_pareto_ratio;
            const pCI = row.ci95_pareto_ratio || [pMean, pMean];
            const nMean = row.mean_nash_welfare;
            const nCI = row.ci95_nash_welfare || [nMean, nMean];

            const pErr: [number, number] = [
              Math.max(0, pMean - pCI[0]),
              Math.max(0, pCI[1] - pMean),
            ];
            const nErr: [number, number] = [
              Math.max(0, nMean - nCI[0]),
              Math.max(0, nCI[1] - nMean),
            ];

            const wilcoxonP = row.wilcoxon_pareto_vs_engine;
            let wPStr = '*(Engine)*';
            if (m !== 'consensus_engine' && wilcoxonP && wilcoxonP.p_value != null) {
              const pVal = wilcoxonP.p_value;
              const sig = pVal < 0.01 ? '**' : pVal < 0.05 ? '*' : 'n.s.';
              wPStr = pVal < 1e-6 ? `p < 1e-6 (${sig})` : `p = ${pVal.toFixed(4)} (${sig})`;
            }

            const wilcoxonN = row.wilcoxon_nash_vs_engine;
            let wNStr = '*(Engine)*';
            if (m !== 'consensus_engine' && wilcoxonN && wilcoxonN.p_value != null) {
              const pVal = wilcoxonN.p_value;
              const sig = pVal < 0.01 ? '**' : pVal < 0.05 ? '*' : 'n.s.';
              wNStr = pVal < 1e-6 ? `p < 1e-6 (${sig})` : `p = ${pVal.toFixed(4)} (${sig})`;
            }

            return {
              name: m,
              label:
                m === 'consensus_engine'
                  ? 'Consensus Engine (Ours)'
                  : m === 'public_midpoint'
                  ? 'Public Range Midpoint'
                  : m === 'naive_average'
                  ? 'Private Ideal Average'
                  : m === 'nash_bargaining'
                  ? 'Nash Bargaining Solution'
                  : m,
              pareto: pMean,
              paretoLow: pCI[0],
              paretoHigh: pCI[1],
              paretoError: pErr,
              nash: nMean,
              nashLow: nCI[0],
              nashHigh: nCI[1],
              nashError: nErr,
              agreement: (row.agreement_rate || 1.0) * 100,
              wilcoxonPareto: wPStr,
              wilcoxonNash: wNStr,
              infoLevel:
                m === 'consensus_engine'
                  ? 'Private Utilities'
                  : m === 'public_midpoint'
                  ? 'Public Bounds'
                  : m === 'naive_average'
                  ? 'Semi-Oracle'
                  : m === 'nash_bargaining'
                  ? 'Full Oracle'
                  : 'Public',
              rounds: row.mean_rounds || 0,
            };
          });

          // Ensure consensus_engine is first
          mapped.sort((a, b) => (a.name === 'consensus_engine' ? -1 : b.name === 'consensus_engine' ? 1 : 0));
          setData(mapped);
        }
      } catch (e) {
        console.warn('Using tracked log benchmarks (backend report fetch failed):', e);
      }
    }
    fetchReport();
  }, []);

  return (
    <div className="space-y-8 w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 overflow-x-hidden">
      {/* Header Banner */}
      <div className="rounded-2xl border border-zinc-800 bg-gradient-to-b from-zinc-900/90 to-zinc-950/90 p-4 sm:p-8 backdrop-blur-xl relative overflow-hidden w-full max-w-full">
        <div className="absolute top-0 right-0 w-96 h-96 bg-emerald-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-emerald-950/60 border border-emerald-800/40 text-emerald-400 text-xs font-mono mb-3">
              <Award className="w-3.5 h-3.5" />
              <span>Scientific Evaluation (N=30 Stage a, Seeds 42–71)</span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-bold text-zinc-100 tracking-tight">
              Empirical Benchmark & Efficiency Frontiers
            </h1>
            <p className="text-sm text-zinc-400 mt-1 max-w-2xl">
              Evaluating multi-agent negotiation with private information against zero-knowledge arithmetic heuristics,
              semi-oracles, and game-theoretic bounds with 95% bootstrap confidence intervals.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2 sm:gap-3 bg-zinc-900/80 border border-zinc-800 p-3 rounded-xl max-w-full">
            <div className="text-center px-3 border-r border-zinc-800">
              <p className="text-xs text-zinc-400">Total Trials</p>
              <p className="text-xl font-bold font-mono text-zinc-100">180</p>
            </div>
            <div className="text-center px-3 border-r border-zinc-800">
              <p className="text-xs text-zinc-400">Bootstrap B</p>
              <p className="text-xl font-bold font-mono text-emerald-400">1,000</p>
            </div>
            <div className="text-center px-3">
              <p className="text-xs text-zinc-400">Breach Rate</p>
              <p className="text-xl font-bold font-mono text-teal-400">0.0%</p>
            </div>
          </div>
        </div>
      </div>

      {/* Honest Scientific Finding Callout */}
      <div className="rounded-xl border border-amber-900/40 bg-amber-950/20 p-4 sm:p-5 flex items-start space-x-4 w-full max-w-full overflow-hidden">
        <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
        <div className="text-xs sm:text-sm text-zinc-300 leading-relaxed min-w-0 flex-1 break-words">
          <span className="font-semibold text-amber-300">What This Benchmark Measures: </span>
          All 30 Consensus Engine trials concluded in round 1 with exactly 4 LLM calls (1 mediator proposal + 3 critiques),
          meaning every party accepted the mediator&apos;s initial proposal. The engine achieves a utilitarian welfare ratio of
          <span className="font-mono text-amber-200"> 0.9156 vs. 0.9082</span> for public midpoint (Wilcoxon{' '}
          <span className="font-mono text-amber-200">p = 0.0017**</span>; Nash welfare <span className="font-mono text-amber-200">p = 0.0015**</span>).
          This reflects the mediator adjusting the midpoint using issue descriptions, NOT iterative back-and-forth negotiation.
        </div>
      </div>

      {/* Section 1: Main Efficiency Bar Charts with Error Bars */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 rounded-2xl border border-zinc-800 bg-zinc-900/50 p-4 sm:p-6 backdrop-blur-sm w-full max-w-full overflow-hidden">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-6">
            <div>
              <h2 className="text-lg font-semibold text-zinc-100 flex items-center space-x-2">
                <Scale className="w-5 h-5 text-emerald-400" />
                <span>Efficiency Metric Comparison with 95% Bootstrap CI</span>
              </h2>
              <p className="text-xs text-zinc-400 mt-0.5">Error bars display empirical 95% confidence intervals from 1,000 bootstrap resamples</p>
            </div>
            <div className="flex bg-zinc-950 p-1 rounded-lg border border-zinc-800 text-xs">
              <button
                onClick={() => setMetricTab('pareto')}
                className={`px-3 py-1 rounded-md font-medium transition-colors ${
                  metricTab === 'pareto' ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' : 'text-zinc-400 hover:text-zinc-200'
                }`}
              >
                Utilitarian Ratio
              </button>
              <button
                onClick={() => setMetricTab('nash')}
                className={`px-3 py-1 rounded-md font-medium transition-colors ${
                  metricTab === 'nash' ? 'bg-teal-500/20 text-teal-400 border border-teal-500/30' : 'text-zinc-400 hover:text-zinc-200'
                }`}
              >
                Nash Welfare
              </button>
            </div>
          </div>

          <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={data}
                margin={{ top: 20, right: 30, left: 0, bottom: 25 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                <XAxis
                  dataKey="label"
                  stroke="#71717a"
                  fontSize={11}
                  tickLine={false}
                  interval={0}
                  angle={-10}
                  textAnchor="end"
                />
                <YAxis
                  stroke="#71717a"
                  fontSize={11}
                  domain={metricTab === 'pareto' ? [0.75, 1.02] : [0.2, 0.6]}
                  tickLine={false}
                />
                <Tooltip
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const d = payload[0].payload as BenchmarkMethod;
                      return (
                        <div className="bg-zinc-950 border border-zinc-800 p-3 rounded-xl shadow-xl text-xs space-y-1">
                          <p className="font-bold text-zinc-100">{d.label}</p>
                          <p className="text-zinc-400">Info Level: <span className="text-zinc-200">{d.infoLevel}</span></p>
                          {metricTab === 'pareto' ? (
                            <>
                              <p className="text-emerald-400 font-mono font-semibold">
                                Utilitarian Ratio: {d.pareto.toFixed(4)} [{d.paretoLow.toFixed(4)}, {d.paretoHigh.toFixed(4)}]
                              </p>
                              <p className="text-zinc-400">Wilcoxon vs Engine: <span className="font-mono text-amber-300">{d.wilcoxonPareto}</span></p>
                            </>
                          ) : (
                            <>
                              <p className="text-teal-400 font-mono font-semibold">
                                Nash Welfare: {d.nash.toFixed(4)} [{d.nashLow.toFixed(4)}, {d.nashHigh.toFixed(4)}]
                              </p>
                              <p className="text-zinc-400">Wilcoxon vs Engine: <span className="font-mono text-amber-300">{d.wilcoxonNash}</span></p>
                            </>
                          )}
                          <p className="text-zinc-500 pt-1">Rounds: {d.rounds}</p>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Bar
                  dataKey={metricTab === 'pareto' ? 'pareto' : 'nash'}
                  fill={metricTab === 'pareto' ? '#10b981' : '#14b8a6'}
                  radius={[6, 6, 0, 0]}
                  barSize={48}
                >
                  <ErrorBar
                    dataKey={metricTab === 'pareto' ? 'paretoError' : 'nashError'}
                    width={10}
                    strokeWidth={2}
                    stroke="#fbbf24"
                  />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Statistical Significance Table Card */}
        <div className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-6 flex flex-col justify-between backdrop-blur-sm">
          <div>
            <h3 className="text-sm font-semibold text-zinc-200 flex items-center space-x-2">
              <Info className="w-4 h-4 text-emerald-400" />
              <span>Two-Sided Wilcoxon Tests (N=30)</span>
            </h3>
            <p className="text-xs text-zinc-400 mt-1">
              Paired Wilcoxon signed-rank test comparing each baseline with Consensus Engine.
            </p>

            <div className="mt-4 space-y-3">
              {data.map((b) => (
                <div key={b.name} className="p-2.5 rounded-lg bg-zinc-950/60 border border-zinc-800/80 text-xs">
                  <div className="flex justify-between items-center mb-1">
                    <span className="font-medium text-zinc-200">{b.label}</span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400 font-mono">
                      {b.infoLevel}
                    </span>
                  </div>
                  <div className="flex justify-between text-zinc-400 text-[11px] font-mono">
                    <span>Welfare:</span>
                    <span className={b.wilcoxonPareto.includes('**') ? 'text-amber-400 font-semibold' : 'text-zinc-300'}>
                      {b.wilcoxonPareto}
                    </span>
                  </div>
                  <div className="flex justify-between text-zinc-400 text-[11px] font-mono">
                    <span>Nash:</span>
                    <span className={b.wilcoxonNash.includes('**') ? 'text-amber-400 font-semibold' : 'text-zinc-300'}>
                      {b.wilcoxonNash}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-zinc-800/80 text-[11px] text-zinc-500">
            ** p &lt; 0.01; * p &lt; 0.05; n.s. indicates statistical tie.
          </div>
        </div>
      </div>

      {/* Section 2: Reservation Impasse & Anchor Manipulation */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Chart A: Reservation Threshold Breach Rate */}
        <div className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-6 backdrop-blur-sm">
          <div className="flex items-center justify-between mb-4">
            <div>
              <div className="flex items-center space-x-2">
                <ShieldCheck className="w-5 h-5 text-rose-400" />
                <h3 className="text-base font-semibold text-zinc-100">Reservation Value Breach Rate</h3>
              </div>
              <p className="text-xs text-zinc-400 mt-1">
                Midpoint averaging produces 100% contract breach when r ≥ 0.70; CE breach is 0.0% by construction.
              </p>
            </div>
            <span className="text-[10px] font-mono bg-zinc-800 text-zinc-400 border border-zinc-700 px-2 py-0.5 rounded-full">
              Deterministic Baselines N=100
            </span>
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={reservationData} margin={{ top: 10, right: 20, left: -10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                <XAxis dataKey="threshold" stroke="#71717a" fontSize={11} tickLine={false} />
                <YAxis stroke="#71717a" fontSize={11} unit="%" tickLine={false} />
                <Tooltip
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const d = payload[0].payload;
                      return (
                        <div className="bg-zinc-950 border border-zinc-800 p-3 rounded-xl shadow-xl text-xs space-y-1">
                          <p className="font-bold text-zinc-200">{d.threshold}</p>
                          <p className="text-rose-400 font-mono">Midpoint Breach Rate: {d.midpointBreach}%</p>
                          <p className="text-emerald-400 font-mono">CE Involuntary Breach: 0.0% (Enforced by code)</p>
                          <p className="text-zinc-400 font-mono">CE Agreement: {d.ceAgree !== null ? `${d.ceAgree}% (${d.ceNote})` : d.ceNote}</p>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Area type="monotone" dataKey="midpointBreach" stroke="#f43f5e" fill="#f43f5e" fillOpacity={0.15} name="Public Midpoint Breach %" />
                <Line type="monotone" dataKey="ceBreach" stroke="#10b981" strokeWidth={3} name="Consensus Engine Breach %" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-2 text-[11px] text-zinc-400 flex justify-between">
            <span>Involuntary breach is strictly 0.0% by construction (`would_accept`)</span>
            <span className="text-emerald-400 font-medium">r=0.40 tested; r≥0.55 unrun</span>
          </div>
        </div>

        {/* Chart B: Strategic Anchor Manipulation & Span Capture */}
        <div className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-6 backdrop-blur-sm">
          <div className="flex items-center justify-between mb-4">
            <div>
              <div className="flex items-center space-x-2">
                <AlertTriangle className="w-5 h-5 text-amber-400" />
                <h3 className="text-base font-semibold text-zinc-100">Strategic Anchor Baseline Arithmetic</h3>
              </div>
              <p className="text-xs text-zinc-400 mt-1">
                Naive average concedes up to 25.0% contract span under anchor inflation; midpoint resists price shift.
              </p>
            </div>
            <span className="text-[10px] font-mono bg-zinc-800 text-zinc-400 border border-zinc-700 px-2 py-0.5 rounded-full">
              Deterministic Arithmetic (No LLM)
            </span>
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={anchorData} margin={{ top: 10, right: 20, left: -10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                <XAxis dataKey="level" stroke="#71717a" fontSize={11} tickLine={false} />
                <YAxis stroke="#71717a" fontSize={11} unit="%" tickLine={false} />
                <Tooltip
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const d = payload[0].payload;
                      return (
                        <div className="bg-zinc-950 border border-zinc-800 p-3 rounded-xl shadow-xl text-xs space-y-1">
                          <p className="font-bold text-zinc-200">Inflation: {d.level} (+${d.stated.toFixed(2)})</p>
                          <p className="text-amber-400 font-mono">Naive Span Captured: +{d.naiveCapture}% (+${d.naiveShift.toFixed(2)})</p>
                          <p className="text-zinc-400 font-mono">Midpoint Span Shift: +${d.midShift.toFixed(2)} (0.0% capture)</p>
                          <p className="text-zinc-500 text-[10px]">CE anchor ablation: Not yet run</p>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Bar dataKey="naiveCapture" fill="#f59e0b" name="Naive Average Capture %" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-2 text-[11px] text-zinc-400 flex justify-between">
            <span>Naive averaging coordinate shifts linearly with stated demand</span>
            <span className="text-zinc-400 font-medium">CE anchor sweep not yet run</span>
          </div>
        </div>
      </div>
    </div>
  );
};
