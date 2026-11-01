'use client';

import React, { useState, useEffect } from 'react';
import {
  Lock,
  Eye,
  AlertOctagon,
  HelpCircle,
  ShieldAlert,
  Cpu,
  Info,
  Terminal,
} from 'lucide-react';
import { listLogs, readLogRecords } from '@/lib/api';

interface AgentLeakage {
  agent: string;
  role: string;
  cosineSim: number;
  klDiv: number;
  weights: {
    issue: string;
    trueWeight: number;
    inferredWeight: number;
  }[];
}

export const PrivacyProbeView: React.FC = () => {
  const [probeData, setProbeData] = useState<{
    meanCosine: number;
    uniformBaseline: number;
    dirichletBaseline: number;
    liftOverUniform: number | null;
    agents: AgentLeakage[];
  } | null>(null);
  const [selectedAgent, setSelectedAgent] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    async function checkLogsForPrivacy() {
      setLoading(true);
      try {
        const fileList = await listLogs();
        let foundPrivacy: any = null;

        for (const f of fileList) {
          const recs = await readLogRecords(f.filename);
          for (const r of recs) {
            if ((r as any).privacy && (r as any).privacy.mean_cosine_similarity) {
              foundPrivacy = (r as any).privacy;
              break;
            }
          }
          if (foundPrivacy) break;
        }

        if (foundPrivacy) {
          const agents: AgentLeakage[] = [];
          if (foundPrivacy.per_agent_leakage) {
            for (const [name, stats] of Object.entries(foundPrivacy.per_agent_leakage as Record<string, any>)) {
              agents.push({
                agent: name,
                role: 'Stakeholder',
                cosineSim: stats.cosine_similarity || 0,
                klDiv: stats.kl_divergence || 0,
                weights: [],
              });
            }
          }
          setProbeData({
            meanCosine: foundPrivacy.mean_cosine_similarity,
            uniformBaseline: foundPrivacy.uniform_baseline_cosine || 0.80,
            dirichletBaseline: foundPrivacy.dirichlet_baseline_cosine || 0.67,
            liftOverUniform: foundPrivacy.lift_over_uniform ?? null,
            agents,
          });
          if (agents.length > 0) setSelectedAgent(agents[0].agent);
        } else {
          setProbeData(null);
        }
      } catch (e) {
        console.warn('Failed to inspect privacy probe logs:', e);
        setProbeData(null);
      } finally {
        setLoading(false);
      }
    }
    checkLogsForPrivacy();
  }, []);

  return (
    <div className="space-y-8 w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 overflow-x-hidden">
      {/* Header Banner */}
      <div className="rounded-2xl border border-zinc-800 bg-gradient-to-b from-zinc-900/90 to-zinc-950/90 p-4 sm:p-8 backdrop-blur-xl relative overflow-hidden w-full max-w-full">
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
              Measuring how much private utility preference geometry leaks through natural-language dialogue to a
              post-hoc reconstruction observer probe that reads only the unedited transcript.
            </p>
          </div>
        </div>
      </div>

      {/* Research Insight Callout */}
      <div className="rounded-xl border border-cyan-900/40 bg-cyan-950/20 p-4 sm:p-5 flex items-start space-x-4 w-full max-w-full overflow-hidden">
        <AlertOctagon className="w-6 h-6 text-cyan-400 shrink-0 mt-0.5" />
        <div className="text-xs sm:text-sm text-zinc-300 leading-relaxed space-y-1 min-w-0 flex-1 break-words">
          <p className="font-semibold text-cyan-300">
            Research Insight: Architectural Isolation Does Not Equal Information Privacy
          </p>
          <p>
            Multi-agent frameworks often claim privacy &ldquo;by design&rdquo; because raw numeric weights and reservation
            thresholds are never transmitted directly in message payloads. However, natural-language bargaining requires
            communicating trade-offs (e.g., &ldquo;I cannot accept 20-day delivery unless price drops&rdquo;), which can
            expose weight priorities to an observer.
          </p>
        </div>
      </div>

      {/* Main Content Area: Empirical Data or Explicit Unrun State */}
      {loading ? (
        <div className="rounded-2xl border border-zinc-800 p-12 text-center text-xs text-zinc-400">
          Inspecting benchmark logs for privacy probe data...
        </div>
      ) : probeData ? (
        <div className="space-y-6">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 font-mono text-center">
            <div className="p-4 rounded-xl bg-zinc-900/60 border border-zinc-800">
              <p className="text-xs text-zinc-400">Mean Cosine Leakage</p>
              <p className="text-2xl font-bold text-rose-400 mt-1">{probeData.meanCosine.toFixed(4)}</p>
            </div>
            <div className="p-4 rounded-xl bg-zinc-900/60 border border-zinc-800">
              <p className="text-xs text-zinc-400">Uniform-Guess Baseline</p>
              <p className="text-2xl font-bold text-zinc-300 mt-1">{probeData.uniformBaseline.toFixed(4)}</p>
            </div>
            <div className="p-4 rounded-xl bg-zinc-900/60 border border-zinc-800">
              <p className="text-xs text-zinc-400">Lift Over Uniform</p>
              <p className="text-2xl font-bold text-amber-400 mt-1">
                {probeData.liftOverUniform != null ? `${(probeData.liftOverUniform * 100).toFixed(1)}%` : '—'}
              </p>
            </div>
          </div>
        </div>
      ) : (
        <div className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-8 sm:p-12 text-center backdrop-blur-sm">
          <Info className="w-10 h-10 text-cyan-400/80 mx-auto mb-4" />
          <h2 className="text-lg font-semibold text-zinc-100">No Empirical Privacy Probe Run Available</h2>
          <p className="text-xs sm:text-sm text-zinc-400 mt-2 max-w-xl mx-auto leading-relaxed">
            The published benchmark run (`results/benchmark_business_deal.jsonl`, N=30) evaluated efficiency and
            fairness without the optional post-hoc LLM reconstruction probe to conserve API token quotas.
          </p>

          <div className="mt-6 p-4 rounded-xl bg-zinc-950 border border-zinc-800 max-w-2xl mx-auto text-left font-mono text-xs">
            <div className="flex items-center space-x-2 text-zinc-400 mb-2 font-sans font-medium">
              <Terminal className="w-4 h-4 text-emerald-400" />
              <span>Command to execute live privacy reconstruction probe:</span>
            </div>
            <pre className="text-emerald-400 overflow-x-auto p-2 bg-zinc-900/80 rounded border border-zinc-800/80">
              python -m src.eval.runner --scenario business_deal --n_trials 30 --protocol single_text --model_config 20b_vs_20b --run_privacy_probe
            </pre>
          </div>

          <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-4 max-w-2xl mx-auto text-left text-xs">
            <div className="p-4 rounded-xl bg-zinc-950/60 border border-zinc-800/80">
              <h3 className="font-semibold text-zinc-200 mb-1">Uniform-Guess Baseline</h3>
              <p className="text-zinc-400 leading-relaxed">
                Guessing equal weights [1/K, ..., 1/K] across 5 issues scores an expected ~0.80 cosine similarity for
                non-negative normalized probability vectors (not orthogonal 0.0).
              </p>
            </div>
            <div className="p-4 rounded-xl bg-zinc-950/60 border border-zinc-800/80">
              <h3 className="font-semibold text-zinc-200 mb-1">Empirical Lift Metric</h3>
              <p className="text-zinc-400 leading-relaxed">
                Leakage is quantified as lift over the uniform baseline: (probe_cosine - uniform_cosine) / uniform_cosine.
                A probe that extracts real dialogue structure scores above the ~0.80 uniform floor.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Methodology Explainer */}
      <div className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-6 backdrop-blur-sm">
        <h3 className="text-sm font-semibold text-zinc-200 flex items-center space-x-2 mb-3">
          <HelpCircle className="w-4 h-4 text-zinc-400" />
          <span>Probe Architecture &amp; Reference Baselines</span>
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-xs text-zinc-400 leading-relaxed">
          <div className="space-y-1">
            <h4 className="font-semibold text-zinc-300">1. Reconstruction Prompt</h4>
            <p>
              An independent probe LLM evaluates the dialogue transcript post-session with zero access to system prompts,
              utility definitions, or internal scratchpads. It attempts to infer each party&apos;s issue weight vector.
            </p>
          </div>
          <div className="space-y-1">
            <h4 className="font-semibold text-zinc-300">2. Reference Baselines</h4>
            <p>
              Because normalized weight vectors live on a simplex with non-negative coordinates, random uniform guessing
              averages ~0.80 cosine similarity (5 issues), and random Dirichlet draws average ~0.65–0.70.
            </p>
          </div>
          <div className="space-y-1">
            <h4 className="font-semibold text-zinc-300">3. Practical Implication</h4>
            <p>
              Measuring lift over uniform guessing tests whether dialogue exchanges expose additional preference priorities
              beyond zero-knowledge geometric priors.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
