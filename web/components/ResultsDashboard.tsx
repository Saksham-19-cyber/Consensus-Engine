'use client';

import React, { useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  CartesianGrid,
  Legend,
  ErrorBar,
  AreaChart,
  Area,
} from 'recharts';
import { Award, AlertTriangle, ShieldCheck, Scale, Info, CheckCircle2 } from 'lucide-react';

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

const benchmarkData: BenchmarkMethod[] = [
  {
    name: 'consensus_engine',
    label: 'Consensus Engine (Ours)',
    pareto: 0.918,
    paretoLow: 0.892,
    paretoHigh: 0.938,
    paretoError: [0.026, 0.02],
    nash: 0.364,
    nashLow: 0.305,
    nashHigh: 0.418,
    nashError: [0.059, 0.054],
    agreement: 100.0,
    wilcoxonPareto: 'Baseline',
    wilcoxonNash: 'Baseline',
    infoLevel: 'Private (Zero Central)',
    rounds: 2.0,
  },
  {
    name: 'public_midpoint',
    label: 'Public Range Midpoint',
    pareto: 0.908,
    paretoLow: 0.895,
    paretoHigh: 0.92,
    paretoError: [0.013, 0.012],
    nash: 0.389,
    nashLow: 0.374,
    nashHigh: 0.402,
    nashError: [0.015, 0.013],
    agreement: 100.0,
    wilcoxonPareto: 'p = 0.281 (n.s.)',
    wilcoxonNash: 'p = 0.312 (n.s.)',
    infoLevel: 'Public Bounds Only',
    rounds: 0.0,
  },
  {
    name: 'private_ideal_average',
    label: 'Private Ideal Average',
    pareto: 0.953,
    paretoLow: 0.943,
    paretoHigh: 0.961,
    paretoError: [0.01, 0.008],
    nash: 0.453,
    nashLow: 0.434,
    nashHigh: 0.472,
    nashError: [0.019, 0.019],
    agreement: 100.0,
    wilcoxonPareto: 'p = 0.004 (**)',
    wilcoxonNash: 'p = 0.002 (**)',
    infoLevel: 'Semi-Oracle',
    rounds: 0.0,
  },
  {
    name: 'nash_bargaining',
    label: 'Nash Bargaining Solution',
    pareto: 0.998,
    paretoLow: 0.996,
    paretoHigh: 0.999,
    paretoError: [0.002, 0.001],
    nash: 0.528,
    nashLow: 0.503,
    nashHigh: 0.554,
    nashError: [0.025, 0.026],
    agreement: 100.0,
    wilcoxonPareto: 'p < 10⁻⁵ (**)',
    wilcoxonNash: 'p < 10⁻⁶ (**)',
    infoLevel: 'Full Omniscient Oracle',
    rounds: 0.0,
  },
];

const reservationData = [
  { threshold: 'r ≤ 0.40', rVal: 0.4, midpointBreach: 0.0, ceBreach: 0.0, midpointAgree: 100.0, ceAgree: 100.0 },
  { threshold: 'r = 0.55', rVal: 0.55, midpointBreach: 4.0, ceBreach: 0.0, midpointAgree: 96.0, ceAgree: 100.0 },
  { threshold: 'r = 0.60', rVal: 0.6, midpointBreach: 33.0, ceBreach: 0.0, midpointAgree: 67.0, ceAgree: 40.0 },
  { threshold: 'r = 0.65', rVal: 0.65, midpointBreach: 78.0, ceBreach: 0.0, midpointAgree: 22.0, ceAgree: 20.0 },
  { threshold: 'r = 0.70', rVal: 0.7, midpointBreach: 100.0, ceBreach: 0.0, midpointAgree: 0.0, ceAgree: 0.0 },
];

const anchorData = [
  { level: '+10%', stated: 9.0, naiveShift: 3.0, naiveCapture: 3.3, ceShift: 0.0, ceCapture: 0.0, flagged: 0 },
  { level: '+25%', stated: 22.5, naiveShift: 7.5, naiveCapture: 8.3, ceShift: -0.42, ceCapture: -0.5, flagged: 100 },
  { level: '+50%', stated: 45.0, naiveShift: 15.0, naiveCapture: 16.7, ceShift: 0.35, ceCapture: 0.4, flagged: 100 },
  { level: '+75%', stated: 67.5, naiveShift: 22.5, naiveCapture: 25.0, ceShift: 0.82, ceCapture: 0.9, flagged: 100 },
];

export const ResultsDashboard: React.FC = () => {
  const [metricTab, setMetricTab] = useState<'pareto' | 'nash'>('pareto');

  return (
    <div className="space-y-8 w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 overflow-x-hidden">
      {/* Header Banner */}
      <div className="rounded-2xl border border-zinc-800 bg-gradient-to-b from-zinc-900/90 to-zinc-950/90 p-4 sm:p-8 backdrop-blur-xl relative overflow-hidden w-full max-w-full">
        <div className="absolute top-0 right-0 w-96 h-96 bg-emerald-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-emerald-950/60 border border-emerald-800/40 text-emerald-400 text-xs font-mono mb-3">
              <Award className="w-3.5 h-3.5" />
              <span>Rigorous Scientific Evaluation (N=30 per cell)</span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-bold text-zinc-100 tracking-tight">
              Empirical Benchmark & Efficiency Frontiers
            </h1>
            <p className="text-sm text-zinc-400 mt-1 max-w-2xl">
              Evaluating multi-agent negotiation with private information against zero-knowledge arithmetic heuristics,
              semi-oracles, and omniscient game-theoretic bounds with 95% bootstrap confidence intervals.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2 sm:gap-3 bg-zinc-900/80 border border-zinc-800 p-3 rounded-xl max-w-full">
            <div className="text-center px-3 border-r border-zinc-800">
              <p className="text-xs text-zinc-400">Total Trials</p>
              <p className="text-xl font-bold font-mono text-zinc-100">120+</p>
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

      {/* Critical Scientific Finding Callout */}
      <div className="rounded-xl border border-amber-900/40 bg-amber-950/20 p-4 sm:p-5 flex items-start space-x-4 w-full max-w-full overflow-hidden">
        <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
        <div className="text-xs sm:text-sm text-zinc-300 leading-relaxed min-w-0 flex-1 break-words">
          <span className="font-semibold text-amber-300">The Limits of Raw Efficiency in Symmetric Spaces: </span>
          Consensus Engine does not outperform simple arithmetic midpoint averaging on raw Pareto efficiency in continuous
          unconstrained spaces (<span className="font-mono text-amber-200">0.918 vs 0.908, p=0.281, n.s.</span>). The true value of
          multi-agent dialogue emerges strictly in non-trivial regimes: preventing contract breaches under strict reservation thresholds
          and neutralizing bad-faith strategic anchor manipulation.
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
              <p className="text-xs text-zinc-400 mt-0.5">Error bars display empirical 95% confidence intervals from 1,000 resamples</p>
            </div>
            <div className="flex bg-zinc-950 p-1 rounded-lg border border-zinc-800 text-xs">
              <button
                onClick={() => setMetricTab('pareto')}
                className={`px-3 py-1 rounded-md font-medium transition-colors ${
                  metricTab === 'pareto' ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' : 'text-zinc-400 hover:text-zinc-200'
                }`}
              >
                Pareto Ratio
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
                data={benchmarkData}
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
                  domain={metricTab === 'pareto' ? [0.8, 1.02] : [0.2, 0.6]}
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
                                Pareto Ratio: {d.pareto.toFixed(3)} [{d.paretoLow.toFixed(3)}, {d.paretoHigh.toFixed(3)}]
                              </p>
                              <p className="text-zinc-400">Wilcoxon vs Engine: <span className="font-mono text-amber-300">{d.wilcoxonPareto}</span></p>
                            </>
                          ) : (
                            <>
                              <p className="text-teal-400 font-mono font-semibold">
                                Nash Welfare: {d.nash.toFixed(3)} [{d.nashLow.toFixed(3)}, {d.nashHigh.toFixed(3)}]
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
              <span>Two-Sided Wilcoxon Inference</span>
            </h3>
            <p className="text-xs text-zinc-400 mt-1">
              Hypothesis test evaluating whether baseline methods significantly differ from Consensus Engine.
            </p>

            <div className="mt-4 space-y-3">
              {benchmarkData.map((b) => (
                <div key={b.name} className="p-2.5 rounded-lg bg-zinc-950/60 border border-zinc-800/80 text-xs">
                  <div className="flex justify-between items-center mb-1">
                    <span className="font-medium text-zinc-200">{b.label}</span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400 font-mono">
                      {b.infoLevel}
                    </span>
                  </div>
                  <div className="flex justify-between text-zinc-400 text-[11px] font-mono">
                    <span>Pareto:</span>
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
            ** Significant at p &lt; 0.01; * Significant at p &lt; 0.05; n.s. indicates statistical tie.
          </div>
        </div>
      </div>

      {/* Section 2: Why Negotiation Matters (Reservation Impasse & Anchor Manipulation) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Chart A: Reservation Threshold Breach Rate */}
        <div className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-6 backdrop-blur-sm">
          <div className="flex items-center justify-between mb-4">
            <div>
              <div className="flex items-center space-x-2">
                <ShieldCheck className="w-5 h-5 text-rose-400" />
                <h3 className="text-base font-semibold text-zinc-100">Contract Breach Under Reservation Thresholds</h3>
              </div>
              <p className="text-xs text-zinc-400 mt-1">
                Midpoint averaging produces 100% contract breach when r ≥ 0.70; Consensus Engine maintains 0% breach.
              </p>
            </div>
            <span className="text-[10px] font-mono bg-rose-950 text-rose-400 border border-rose-800/40 px-2 py-0.5 rounded-full">
              N=100 Trials
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
                          <p className="text-emerald-400 font-mono">Consensus Engine Breach: 0.0% (Guaranteed)</p>
                          <p className="text-zinc-400">CE Voluntary Agreement: {d.ceAgree}% (Rem. Safe Impasse)</p>
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
            <span>Mean agent min utility under midpoint: <strong className="text-zinc-200 font-mono">0.6168</strong></span>
            <span className="text-emerald-400 font-medium">Consensus Engine Breach: 0.0% across all rows</span>
          </div>
        </div>

        {/* Chart B: Strategic Anchor Manipulation & Span Capture */}
        <div className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-6 backdrop-blur-sm">
          <div className="flex items-center justify-between mb-4">
            <div>
              <div className="flex items-center space-x-2">
                <AlertTriangle className="w-5 h-5 text-cyan-400" />
                <h3 className="text-base font-semibold text-zinc-100">Strategic Anchor Manipulation Resistance</h3>
              </div>
              <p className="text-xs text-zinc-400 mt-1">
                Naive average concedes 25.0% contract span; Consensus Engine holds bluffer capture beneath 0.9%.
              </p>
            </div>
            <span className="text-[10px] font-mono bg-cyan-950 text-cyan-400 border border-cyan-800/40 px-2 py-0.5 rounded-full">
              N=30 per tier
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
                          <p className="text-emerald-400 font-mono">CE Span Captured: {d.ceCapture > 0 ? `+${d.ceCapture}%` : `${d.ceCapture}%`} (${d.ceShift.toFixed(2)})</p>
                          <p className="text-cyan-400 font-mono">Bluff Flag Sensitivity: {d.flagged}%</p>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
                <Bar dataKey="naiveCapture" fill="#f59e0b" name="Naive Average Capture %" radius={[4, 4, 0, 0]} />
                <Bar dataKey="ceCapture" fill="#10b981" name="Consensus Engine Capture %" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-2 text-[11px] text-zinc-400 flex justify-between">
            <span>Bluff detector triggered at <strong className="text-zinc-200 font-mono">100.0% sensitivity</strong> for inflation ≥ 25%</span>
            <span className="text-cyan-400 font-medium">Mediator throttles concessions</span>
          </div>
        </div>
      </div>
    </div>
  );
};
