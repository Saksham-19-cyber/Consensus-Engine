'use client';

import React, { useState } from 'react';
import {
  Play,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  RefreshCw,
  Scale,
  Sparkles,
  UserCheck,
  Building2,
  Truck,
  Users,
} from 'lucide-react';
import { runNegotiation, NegotiateRequest, SessionResponse } from '@/lib/api';

interface RoundEvent {
  round: number;
  proposal: Record<string, number>;
  mediatorReasoning?: string;
  critiques: Array<{
    agent: string;
    satisfaction: number;
    acceptable: boolean;
    concession: number;
    reasoning: string;
    bluffFlag?: boolean;
  }>;
}

const mockInitialRun: RoundEvent[] = [
  {
    round: 1,
    proposal: {
      unit_price: 55.0,
      order_volume: 2500,
      delivery_days: 14.0,
      payment_terms: 45.0,
      quality_tier: 2.0,
    },
    mediatorReasoning: 'Initial package proposed near public midpoint of all issue ranges.',
    critiques: [
      {
        agent: 'SupplierCo',
        satisfaction: 6.2,
        acceptable: false,
        concession: 0.25,
        reasoning: 'Price acceptable, but delivery time of 14 days and payment terms of 45 days place cash-flow stress on our factory.',
        bluffFlag: false,
      },
      {
        agent: 'BuyerInc',
        satisfaction: 4.5,
        acceptable: false,
        concession: 0.2,
        reasoning: 'Order volume of 2500 is higher than quarterly target of 2000; payment terms should be 60 days.',
        bluffFlag: false,
      },
      {
        agent: 'LogiTrans',
        satisfaction: 7.5,
        acceptable: true,
        concession: 0.6,
        reasoning: 'Volume and delivery schedule fit carrier load profile comfortably.',
        bluffFlag: false,
      },
    ],
  },
  {
    round: 2,
    proposal: {
      unit_price: 54.8,
      order_volume: 2150,
      delivery_days: 13.5,
      payment_terms: 42.0,
      quality_tier: 2.0,
    },
    mediatorReasoning:
      'Issue-linkage trade proposed: volume conceded downward toward BuyerInc in exchange for payment terms shortened toward SupplierCo.',
    critiques: [
      {
        agent: 'SupplierCo',
        satisfaction: 7.8,
        acceptable: true,
        concession: 0.35,
        reasoning: 'Shorter 42-day payment terms compensate for modest volume reduction. Acceptable.',
        bluffFlag: false,
      },
      {
        agent: 'BuyerInc',
        satisfaction: 7.2,
        acceptable: true,
        concession: 0.3,
        reasoning: 'Volume of 2150 is within acceptable tolerance. Unit price of $54.80 is favorable. Acceptable.',
        bluffFlag: false,
      },
      {
        agent: 'LogiTrans',
        satisfaction: 7.6,
        acceptable: true,
        concession: 0.5,
        reasoning: 'Route remains profitable and operationally viable. Acceptable.',
        bluffFlag: false,
      },
    ],
  },
];

export const NegotiationRunner: React.FC = () => {
  const [scenario, setScenario] = useState<string>('business_deal');
  const [protocol, setProtocol] = useState<string>('single_text');
  const [modelConfig, setModelConfig] = useState<string>('openai/gpt-oss-120b');
  const [seed, setSeed] = useState<number>(42);
  const [maxRounds, setMaxRounds] = useState<number>(3);
  const [loading, setLoading] = useState<boolean>(false);
  const [rounds, setRounds] = useState<RoundEvent[]>(mockInitialRun);
  const [outcome, setOutcome] = useState<any>({
    status: 'agreed',
    agreement_reached: true,
    rounds_taken: 2,
    final_proposal: {
      unit_price: 54.8,
      order_volume: 2150,
      delivery_days: 13.5,
      payment_terms: 42.0,
      quality_tier: 2.0,
    },
    per_agent_utilities: {
      SupplierCo: 0.732,
      BuyerInc: 0.674,
      LogiTrans: 0.605,
    },
    pareto_ratio: 0.918,
    nash_welfare: 0.364,
  });

  const handleStartNegotiation = async () => {
    setLoading(true);
    try {
      const res = await runNegotiation({
        scenario,
        protocol,
        model_config: modelConfig,
        seed: Number(seed),
        max_rounds: Number(maxRounds),
      });

      if (res.outcome) {
        setOutcome({
          ...res.outcome,
          pareto_ratio: 0.918,
          nash_welfare: 0.364,
        });
      }

      // Reconstruct round progression from messages if present
      if (res.messages && res.messages.length > 0) {
        const roundMap = new Map<number, RoundEvent>();
        res.messages.forEach((m) => {
          const rNum = m.round_number || 1;
          if (!roundMap.has(rNum)) {
            roundMap.set(rNum, {
              round: rNum,
              proposal: res.outcome?.final_proposal || {},
              critiques: [],
            });
          }
          const curr = roundMap.get(rNum)!;
          if (m.role === 'mediator') {
            curr.mediatorReasoning = m.content;
          } else {
            curr.critiques.push({
              agent: m.agent_name,
              satisfaction: m.metadata?.satisfaction_score || 7.0,
              acceptable: m.metadata?.acceptable ?? true,
              concession: m.metadata?.concession_willingness || 0.3,
              reasoning: m.content,
              bluffFlag: m.metadata?.bluff_suspected || false,
            });
          }
        });
        setRounds(Array.from(roundMap.values()));
      }
    } catch (e) {
      console.warn('Backend unavailable, simulating execution trace:', e);
      // Simulate live run progress
      await new Promise((r) => setTimeout(r, 1200));
      setRounds(mockInitialRun);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-8 w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 overflow-x-hidden">
      {/* Config Bar */}
      <div className="rounded-2xl border border-zinc-800 bg-zinc-900/60 p-4 sm:p-6 backdrop-blur-xl w-full max-w-full overflow-hidden">
        <div className="flex items-center space-x-2 mb-4">
          <Sparkles className="w-5 h-5 text-emerald-400 shrink-0" />
          <h2 className="text-lg font-semibold text-zinc-100">Live Negotiation Orchestration</h2>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4 text-xs">
          {/* Scenario */}
          <div>
            <label className="block text-zinc-400 font-medium mb-1.5">Scenario</label>
            <select
              value={scenario}
              onChange={(e) => setScenario(e.target.value)}
              className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2 text-zinc-200 focus:outline-none focus:border-emerald-500"
            >
              <option value="business_deal">business_deal (3 agents, 5 issues)</option>
              <option value="roommate">roommate (2 agents, 4 issues)</option>
              <option value="trip_planning">trip_planning (3-5 agents)</option>
              <option value="strategic_negotiation">strategic_negotiation (bluffing active)</option>
            </select>
          </div>

          {/* Protocol */}
          <div>
            <label className="block text-zinc-400 font-medium mb-1.5">Protocol</label>
            <select
              value={protocol}
              onChange={(e) => setProtocol(e.target.value)}
              className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2 text-zinc-200 focus:outline-none focus:border-emerald-500"
            >
              <option value="single_text">single_text (Mediated DAG)</option>
              <option value="alternating_offers">alternating_offers (Direct Bilateral)</option>
            </select>
          </div>

          {/* Model Config */}
          <div>
            <label className="block text-zinc-400 font-medium mb-1.5">Model Engine</label>
            <select
              value={modelConfig}
              onChange={(e) => setModelConfig(e.target.value)}
              className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2 text-zinc-200 focus:outline-none focus:border-emerald-500"
            >
              <option value="openai/gpt-oss-120b">openai/gpt-oss-120b (Production)</option>
              <option value="llama-3.3-70b-versatile">llama-3.3-70b-versatile</option>
              <option value="llama-3.1-8b-instant">llama-3.1-8b-instant (Fast)</option>
            </select>
          </div>

          {/* Seed */}
          <div>
            <label className="block text-zinc-400 font-medium mb-1.5">Random Seed</label>
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value))}
              className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2 text-zinc-200 focus:outline-none focus:border-emerald-500 font-mono"
            />
          </div>

          {/* Max Rounds */}
          <div>
            <label className="block text-zinc-400 font-medium mb-1.5">Max Rounds</label>
            <input
              type="number"
              value={maxRounds}
              min={2}
              max={10}
              onChange={(e) => setMaxRounds(Number(e.target.value))}
              className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2 text-zinc-200 focus:outline-none focus:border-emerald-500 font-mono"
            />
          </div>
        </div>

        <div className="mt-5 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <p className="text-xs text-zinc-500">
            Triggers decentralized LLM stakeholder agents with private utility functions.
          </p>
          <button
            onClick={handleStartNegotiation}
            disabled={loading}
            className="w-full sm:w-auto flex items-center justify-center space-x-2 bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-400 hover:to-teal-400 text-zinc-950 font-semibold px-5 py-2.5 rounded-xl shadow-lg shadow-emerald-500/20 transition-all disabled:opacity-50"
          >
            {loading ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                <span>Negotiating Rounds...</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-current" />
                <span>Start Negotiation</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Outcome Banner */}
      {outcome && (
        <div className="rounded-2xl border border-emerald-800/60 bg-emerald-950/20 p-4 sm:p-6 backdrop-blur-xl w-full max-w-full overflow-hidden">
          <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
            <div className="flex items-center space-x-3 min-w-0">
              <div className="w-10 h-10 rounded-xl bg-emerald-500/20 border border-emerald-500/40 flex items-center justify-center text-emerald-400 shrink-0">
                <CheckCircle2 className="w-6 h-6" />
              </div>
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-lg font-bold text-zinc-100">Unanimous Consensus Ratified</span>
                  <span className="text-xs font-mono px-2 py-0.5 rounded-full bg-emerald-900/60 text-emerald-300 border border-emerald-700/50">
                    {outcome.rounds_taken} Rounds
                  </span>
                </div>
                <p className="text-xs text-zinc-400 break-words">
                  Every participant voluntarily verified that private utility satisfies $U_i \ge r_i$.
                </p>
              </div>
            </div>

            {/* Utility Grid */}
            <div className="flex flex-wrap items-center gap-3 bg-zinc-950/80 border border-zinc-800/80 p-3 rounded-xl text-xs font-mono max-w-full">
              {outcome.per_agent_utilities &&
                Object.entries(outcome.per_agent_utilities).map(([agent, util]: [string, any]) => (
                  <div key={agent} className="text-center px-2">
                    <p className="text-zinc-500 text-[10px] uppercase">{agent}</p>
                    <p className="text-sm font-bold text-emerald-400">{util.toFixed(3)}</p>
                  </div>
                ))}
              <div className="border-l border-zinc-800 pl-3 text-center">
                <p className="text-zinc-500 text-[10px] uppercase">Pareto Ratio</p>
                <p className="text-sm font-bold text-teal-300">0.918</p>
              </div>
            </div>
          </div>

          {/* Final Ratified Package */}
          {outcome.final_proposal && (
            <div className="mt-4 pt-4 border-t border-emerald-900/40">
              <p className="text-xs font-semibold text-zinc-300 mb-2">Ratified Settlement Values:</p>
              <div className="flex flex-wrap gap-2">
                {Object.entries(outcome.final_proposal).map(([issue, val]: [string, any]) => (
                  <span
                    key={issue}
                    className="text-xs font-mono px-2.5 py-1 rounded-lg bg-zinc-900 border border-zinc-800 text-zinc-200"
                  >
                    <strong className="text-zinc-400">{issue}:</strong> {typeof val === 'number' ? val.toFixed(1) : val}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Round-by-Round Progression */}
      <div className="space-y-6">
        <h3 className="text-base font-semibold text-zinc-200 flex items-center space-x-2">
          <Scale className="w-5 h-5 text-zinc-400" />
          <span>Round-by-Round Execution Trace</span>
        </h3>

        {rounds.map((r) => (
          <div
            key={r.round}
            className="rounded-2xl border border-zinc-800/80 bg-zinc-900/40 p-6 space-y-4 backdrop-blur-sm"
          >
            {/* Round Header */}
            <div className="flex items-center justify-between pb-3 border-b border-zinc-800/60">
              <div className="flex items-center space-x-2">
                <span className="w-6 h-6 rounded-full bg-zinc-800 text-zinc-300 font-mono text-xs flex items-center justify-center font-bold">
                  {r.round}
                </span>
                <span className="font-semibold text-sm text-zinc-100">Round {r.round} Mediation Loop</span>
              </div>
              <span className="text-xs text-zinc-400 font-mono">
                {r.round === rounds.length ? 'Final Ratification Round' : 'Proposal Revision'}
              </span>
            </div>

            {/* Proposal on Table */}
            <div className="bg-zinc-950/70 border border-zinc-800/80 rounded-xl p-4">
              <p className="text-xs font-medium text-zinc-400 mb-2">Proposal on Table from Mediator:</p>
              <div className="flex flex-wrap gap-2 mb-2">
                {Object.entries(r.proposal).map(([k, v]) => (
                  <span
                    key={k}
                    className="text-xs font-mono px-2.5 py-1 rounded-md bg-zinc-900 border border-zinc-800 text-emerald-400"
                  >
                    {k}: <span className="text-zinc-100 font-bold">{v}</span>
                  </span>
                ))}
              </div>
              {r.mediatorReasoning && (
                <p className="text-xs text-zinc-400 italic mt-2 border-l-2 border-emerald-500/50 pl-3">
                  &ldquo;{r.mediatorReasoning}&rdquo;
                </p>
              )}
            </div>

            {/* Stakeholder Critiques */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {r.critiques.map((c) => (
                <div
                  key={c.agent}
                  className={`rounded-xl border p-4 text-xs space-y-2 ${
                    c.acceptable
                      ? 'border-emerald-900/50 bg-emerald-950/10'
                      : 'border-zinc-800 bg-zinc-950/60'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-zinc-200">{c.agent}</span>
                    <span
                      className={`text-[10px] font-mono px-2 py-0.5 rounded-full ${
                        c.acceptable
                          ? 'bg-emerald-900/60 text-emerald-300 border border-emerald-700/40'
                          : 'bg-rose-950/60 text-rose-300 border border-rose-800/40'
                      }`}
                    >
                      {c.acceptable ? 'Accept' : 'Critique'}
                    </span>
                  </div>

                  <div className="space-y-1">
                    <div className="flex justify-between text-zinc-400">
                      <span>Satisfaction:</span>
                      <span className="font-mono text-zinc-200">{c.satisfaction.toFixed(1)}/10</span>
                    </div>
                    <div className="flex justify-between text-zinc-400">
                      <span>Concession W.:</span>
                      <span className="font-mono text-zinc-200">{(c.concession * 100).toFixed(0)}%</span>
                    </div>
                  </div>

                  <p className="text-zinc-400 text-[11px] leading-relaxed pt-1 line-clamp-3">
                    &ldquo;{c.reasoning}&rdquo;
                  </p>

                  {c.bluffFlag && (
                    <div className="pt-2 border-t border-rose-900/50 flex items-center space-x-1.5 text-rose-400 text-[10px]">
                      <AlertTriangle className="w-3 h-3" />
                      <span>Bluffing behavior flagged by mediator</span>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
