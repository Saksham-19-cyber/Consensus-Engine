'use client';

/**
 * ScenarioBuilder.tsx
 * ─────────────────────────────────────────────────────────────────────────────
 * Free-form natural-language scenario input for Consensus Engine.
 *
 * SCOPE NOTICE — displayed prominently in the UI:
 * Results from free-form scenarios are illustrative only and are NOT covered
 * by the N=30 statistical benchmarks documented in the README
 * (business_deal, roommate, trip_planning, strategic_negotiation).
 *
 * Flow: Input → Parse → Confirm → Run → Results
 */

import React, { useState } from 'react';
import {
  FlaskConical,
  AlertTriangle,
  ArrowLeft,
  ArrowRight,
  CheckCircle2,
  XCircle,
  RefreshCw,
  Play,
  Info,
  Sparkles,
  User,
  Bot,
  Scale,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import {
  API_BASE,
  parseScenario,
  runNegotiation,
  ParseScenarioResponse,
  ParsedStakeholder,
  ParsedIssue,
  SessionResponse,
} from '@/lib/api';

// ─────────────────────────────────────────────────────────────────────────────
// Sub-components
// ─────────────────────────────────────────────────────────────────────────────

/** Prominent amber banner shown on every screen in this mode */
const ExploratorBanner: React.FC<{ compact?: boolean }> = ({ compact }) => (
  <div
    id="exploratory-mode-banner"
    className={`flex items-start gap-3 rounded-xl border border-amber-700/60 bg-amber-950/30 px-4 w-full max-w-full overflow-hidden ${compact ? 'py-2.5' : 'py-4'}`}
  >
    <AlertTriangle className="w-5 h-5 text-amber-400 mt-0.5 shrink-0" />
    <div className="min-w-0 flex-1 break-words">
      <p className="text-sm font-semibold text-amber-300">
        Exploratory Mode — Results Are Not Statistically Validated
      </p>
      {!compact && (
        <p className="text-xs text-amber-400/80 mt-1 leading-relaxed">
          Free-form scenarios are <strong>not</strong> covered by this project&apos;s N=30
          statistical benchmarks. The Pareto efficiency ratios, agreement rates, and Wilcoxon test
          results in the Benchmark Dashboard apply only to the four fixed scenarios
          (business_deal, roommate, trip_planning, strategic_negotiation). Results here are
          illustrative and should not be cited as evidence of the engine&apos;s negotiation
          performance.
        </p>
      )}
    </div>
  </div>
);

/** Source badge: green for user-specified, amber for LLM-inferred */
const SourceBadge: React.FC<{ source: string }> = ({ source }) => {
  const isUser = source === 'user_specified';
  return (
    <span
      className={`inline-flex items-center gap-1 text-[10px] font-mono px-2 py-0.5 rounded-full border ${
        isUser
          ? 'bg-emerald-950/60 text-emerald-300 border-emerald-700/50'
          : 'bg-amber-950/60 text-amber-300 border-amber-700/50'
      }`}
    >
      {isUser ? <User className="w-2.5 h-2.5" /> : <Bot className="w-2.5 h-2.5" />}
      {isUser ? 'User-specified' : 'LLM-inferred'}
    </span>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// State 1: Input
// ─────────────────────────────────────────────────────────────────────────────

interface InputScreenProps {
  description: string;
  seed: number;
  loading: boolean;
  error: string | null;
  onDescriptionChange: (v: string) => void;
  onSeedChange: (v: number) => void;
  onParse: () => void;
}

const InputScreen: React.FC<InputScreenProps> = ({
  description,
  seed,
  loading,
  error,
  onDescriptionChange,
  onSeedChange,
  onParse,
}) => (
  <div className="space-y-6 max-w-4xl mx-auto w-full">
    <ExploratorBanner />

    <div className="rounded-2xl border border-zinc-800 bg-zinc-900/60 p-4 sm:p-6 backdrop-blur-xl space-y-4 w-full max-w-full overflow-hidden">
      <div className="flex items-center gap-2 mb-2">
        <FlaskConical className="w-5 h-5 text-amber-400 shrink-0" />
        <h2 className="text-lg font-semibold text-zinc-100">Describe Your Negotiation</h2>
      </div>

      <p className="text-xs text-zinc-400 leading-relaxed">
        Write a plain-language description of the negotiation situation — who the parties are,
        what they&apos;re negotiating over, and (optionally) each party&apos;s priorities or
        constraints. The more detail you provide for a party&apos;s preferences, the less the
        system has to infer. Any party you don&apos;t fully specify will be labeled{' '}
        <span className="text-amber-300 font-mono">LLM-inferred</span> in the confirmation step.
      </p>

      <div className="w-full max-w-full">
        <label
          htmlFor="scenario-description"
          className="block text-xs font-medium text-zinc-400 mb-1.5"
        >
          Scenario Description
        </label>
        <textarea
          id="scenario-description"
          value={description}
          onChange={(e) => onDescriptionChange(e.target.value)}
          rows={10}
          placeholder={`Example:\n\nA software startup (FounderCo) is negotiating a licensing deal with a large enterprise (EnterpriseCorp). The key issues are: license fee (annual, $50k–$500k), contract duration (1–5 years), support level (basic email vs. dedicated account manager), and data usage rights (whether EnterpriseCorp can use anonymized data for their own analytics).\n\nFounderCo really needs revenue and would accept a lower fee for a longer contract. EnterpriseCorp cares most about data rights and support.`}
          className="w-full max-w-full box-border bg-zinc-950/80 border border-zinc-800 rounded-xl px-4 py-3 text-sm text-zinc-200 placeholder-zinc-600 focus:outline-none focus:border-amber-500/60 focus:ring-1 focus:ring-amber-500/20 resize-y leading-relaxed font-mono whitespace-pre-wrap break-words"
        />
        <p className="text-[11px] text-zinc-600 mt-1">
          Min 20 characters. 2–6 quantifiable issues will be extracted. More than 6 issues will be
          rejected (Pareto search becomes infeasible).
        </p>
      </div>

      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pt-2 w-full">
        <div className="w-full sm:w-32">
          <label htmlFor="scenario-seed" className="block text-xs font-medium text-zinc-400 mb-1.5">
            Random Seed
          </label>
          <input
            id="scenario-seed"
            type="number"
            value={seed}
            onChange={(e) => onSeedChange(Number(e.target.value))}
            className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2 text-zinc-200 text-xs focus:outline-none focus:border-amber-500/60 font-mono"
          />
        </div>
        <button
          id="parse-scenario-btn"
          onClick={onParse}
          disabled={loading || description.trim().length < 20}
          className="w-full sm:w-auto flex items-center justify-center gap-2 bg-gradient-to-r from-amber-500 to-orange-500 hover:from-amber-400 hover:to-orange-400 disabled:opacity-40 text-zinc-950 font-semibold px-6 py-2.5 rounded-xl shadow-lg shadow-amber-500/20 transition-all"
        >
          {loading ? (
            <>
              <RefreshCw className="w-4 h-4 animate-spin" />
              <span>Parsing…</span>
            </>
          ) : (
            <>
              <Sparkles className="w-4 h-4" />
              <span>Parse Scenario</span>
            </>
          )}
        </button>
      </div>

      {error && (
        <div className="flex items-start gap-2 rounded-xl border border-rose-800/50 bg-rose-950/20 px-4 py-3">
          <XCircle className="w-4 h-4 text-rose-400 mt-0.5 shrink-0" />
          <p className="text-xs text-rose-300">{error}</p>
        </div>
      )}
    </div>
  </div>
);

// ─────────────────────────────────────────────────────────────────────────────
// State 2: Confirm
// ─────────────────────────────────────────────────────────────────────────────

interface ConfirmScreenProps {
  parsed: ParseScenarioResponse;
  maxRounds: number;
  running: boolean;
  onMaxRoundsChange: (v: number) => void;
  onConfirm: () => void;
  onBack: () => void;
}

const ConfirmScreen: React.FC<ConfirmScreenProps> = ({
  parsed,
  maxRounds,
  running,
  onMaxRoundsChange,
  onConfirm,
  onBack,
}) => {
  const [showWeights, setShowWeights] = useState<Record<string, boolean>>({});

  const toggleWeights = (name: string) => {
    setShowWeights((prev) => ({ ...prev, [name]: !prev[name] }));
  };

  const inferredCount = parsed.stakeholders.filter((s) => s.source === 'llm_inferred').length;

  return (
    <div className="space-y-6 max-w-5xl mx-auto w-full">
      <ExploratorBanner compact />

      {/* Header */}
      <div className="flex flex-wrap items-center gap-3">
        <button
          id="confirm-back-btn"
          onClick={onBack}
          className="flex items-center gap-1.5 text-xs text-zinc-400 hover:text-zinc-200 transition-colors shrink-0"
        >
          <ArrowLeft className="w-4 h-4" />
          Edit Description
        </button>
        <div className="hidden sm:block flex-1 h-px bg-zinc-800" />
        <span className="text-xs text-zinc-500">Step 2 of 3 — Confirm Parsed Scenario</span>
      </div>

      {/* LLM-inferred warning */}
      {inferredCount > 0 && (
        <div className="flex items-start gap-3 rounded-xl border border-amber-700/40 bg-amber-950/20 px-4 py-3">
          <Bot className="w-4 h-4 text-amber-400 mt-0.5 shrink-0" />
          <p className="text-xs text-amber-300 leading-relaxed">
            <strong>{inferredCount} party preference{inferredCount > 1 ? 's were' : ' was'} generated by the LLM</strong>{' '}
            because your description didn&apos;t fully specify {inferredCount > 1 ? 'their' : 'their'} utility weights,
            ideal values, or reservation value. These are plausible estimates, not user-provided data.
            Review them carefully before confirming.
          </p>
        </div>
      )}

      {/* Issues table */}
      <div className="rounded-2xl border border-zinc-800 bg-zinc-900/60 p-5 space-y-3">
        <h3 className="text-sm font-semibold text-zinc-100 flex items-center gap-2">
          <Scale className="w-4 h-4 text-zinc-400" />
          Parsed Issues ({parsed.issues.length})
          {parsed.pareto_mode === 'monte_carlo' && (
            <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-amber-950/60 text-amber-300 border border-amber-700/50">
              Pareto: Monte Carlo (approximate)
            </span>
          )}
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-zinc-800">
                <th className="text-left text-zinc-500 font-medium pb-2 pr-4">Issue</th>
                <th className="text-left text-zinc-500 font-medium pb-2 pr-4">Range</th>
                <th className="text-left text-zinc-500 font-medium pb-2">What the values mean</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800/60">
              {parsed.issues.map((issue) => (
                <tr key={issue.name}>
                  <td className="py-2.5 pr-4 font-mono text-emerald-400 font-semibold">
                    {issue.name}
                  </td>
                  <td className="py-2.5 pr-4 font-mono text-zinc-300 whitespace-nowrap">
                    [{issue.min_value} – {issue.max_value}]
                  </td>
                  <td className="py-2.5 text-zinc-400 leading-relaxed">{issue.description}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Stakeholders */}
      <div className="rounded-2xl border border-zinc-800 bg-zinc-900/60 p-5 space-y-4">
        <h3 className="text-sm font-semibold text-zinc-100 flex items-center gap-2">
          <User className="w-4 h-4 text-zinc-400" />
          Negotiating Parties ({parsed.stakeholders.length})
        </h3>
        <div className="space-y-3">
          {parsed.stakeholders.map((sh) => (
            <div
              key={sh.name}
              id={`stakeholder-card-${sh.name}`}
              className={`rounded-xl border p-4 space-y-2 w-full max-w-full overflow-hidden ${
                sh.source === 'llm_inferred'
                  ? 'border-amber-800/40 bg-amber-950/10'
                  : 'border-zinc-800 bg-zinc-950/40'
              }`}
            >
              <div className="flex flex-wrap items-center gap-2 sm:gap-3">
                <span className="font-semibold text-zinc-100 text-sm">{sh.name}</span>
                <span className="text-xs text-zinc-500">{sh.role}</span>
                <SourceBadge source={sh.source} />
                <span className="sm:ml-auto text-xs text-zinc-500 font-mono">
                  Reservation ≥ {sh.reservation_value.toFixed(2)}
                </span>
              </div>
              <p className="text-[11px] text-zinc-400 italic leading-relaxed">{sh.persona}</p>

              {/* Collapsible weights */}
              <button
                id={`toggle-weights-${sh.name}`}
                onClick={() => toggleWeights(sh.name)}
                className="flex items-center gap-1 text-[11px] text-zinc-500 hover:text-zinc-300 transition-colors"
              >
                {showWeights[sh.name] ? (
                  <ChevronUp className="w-3 h-3" />
                ) : (
                  <ChevronDown className="w-3 h-3" />
                )}
                {showWeights[sh.name] ? 'Hide' : 'Show'} utility weights &amp; ideal values
              </button>

              {showWeights[sh.name] && (
                <div className="mt-2 overflow-x-auto">
                  <table className="text-[11px] w-full">
                    <thead>
                      <tr className="border-b border-zinc-800">
                        <th className="text-left text-zinc-600 font-medium pb-1.5 pr-3">Issue</th>
                        <th className="text-right text-zinc-600 font-medium pb-1.5 pr-3">Weight</th>
                        <th className="text-right text-zinc-600 font-medium pb-1.5">Ideal value</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-zinc-800/40">
                      {Object.entries(sh.weights).map(([iss, w]) => (
                        <tr key={iss}>
                          <td className="py-1 pr-3 font-mono text-zinc-400">{iss}</td>
                          <td className="py-1 pr-3 text-right font-mono text-zinc-300">
                            {(w * 100).toFixed(1)}%
                          </td>
                          <td className="py-1 text-right font-mono text-emerald-400">
                            {sh.ideal_values[iss]?.toFixed(2) ?? '—'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Field notes & warnings */}
      {(parsed.field_notes.length > 0 || parsed.warnings.length > 0) && (
        <div className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-4 space-y-2">
          <div className="flex items-center gap-2 mb-2">
            <Info className="w-4 h-4 text-zinc-400" />
            <h4 className="text-xs font-semibold text-zinc-300">Parser Notes</h4>
          </div>
          {parsed.field_notes.map((note, i) => (
            <p key={`fn-${i}`} className="text-[11px] text-zinc-400 leading-relaxed">{note}</p>
          ))}
          {parsed.warnings.map((w, i) => (
            <p key={`w-${i}`} className="text-[11px] text-amber-400/80 leading-relaxed">
              ⚠ {w}
            </p>
          ))}
        </div>
      )}

      {/* Confirm & Run */}
      <div className="rounded-2xl border border-zinc-700/60 bg-zinc-900/80 p-5 space-y-4">
        <div className="rounded-xl border border-amber-700/50 bg-amber-950/30 px-4 py-3">
          <p className="text-xs text-amber-300 font-semibold text-center">
            ⚠ Free-form scenarios are not covered by this project&apos;s N=30 statistical
            benchmarks; results here are illustrative, not validated.
          </p>
        </div>

        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="w-full sm:w-auto">
            <label
              htmlFor="confirm-max-rounds"
              className="block text-xs font-medium text-zinc-400 mb-1"
            >
              Max Rounds
            </label>
            <input
              id="confirm-max-rounds"
              type="number"
              value={maxRounds}
              min={2}
              max={10}
              onChange={(e) => onMaxRoundsChange(Number(e.target.value))}
              className="w-full sm:w-24 bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2 text-zinc-200 text-xs focus:outline-none focus:border-emerald-500 font-mono"
            />
          </div>
          <button
            id="confirm-run-btn"
            onClick={onConfirm}
            disabled={running}
            className="w-full sm:w-auto flex items-center justify-center gap-2 bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-400 hover:to-teal-400 disabled:opacity-40 text-zinc-950 font-semibold px-6 py-2.5 rounded-xl shadow-lg shadow-emerald-500/20 transition-all"
          >
            {running ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                <span>Negotiating…</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-current" />
                <span>Confirm &amp; Run Negotiation</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// State 3: Results
// ─────────────────────────────────────────────────────────────────────────────

interface ResultsScreenProps {
  result: SessionResponse;
  parsed: ParseScenarioResponse;
  onReset: () => void;
}

const ResultsScreen: React.FC<ResultsScreenProps> = ({ result, parsed, onReset }) => {
  const outcome = result.outcome;
  const agreed = outcome?.agreement_reached;

  const messages = result.messages || [];
  const roundMap = new Map<
    number,
    { round: number; mediatorReasoning?: string; critiques: any[] }
  >();
  messages.forEach((m) => {
    const rNum = m.round_number || 1;
    if (!roundMap.has(rNum)) {
      roundMap.set(rNum, { round: rNum, critiques: [] });
    }
    const curr = roundMap.get(rNum)!;
    if (m.role === 'mediator') {
      curr.mediatorReasoning = m.content;
    } else {
      curr.critiques.push({
        agent: m.agent_name,
        satisfaction: m.metadata?.satisfaction ?? 7.0,
        acceptable: m.metadata?.acceptable ?? true,
        concession: m.metadata?.concession_willingness ?? 0.3,
        reasoning: m.content,
      });
    }
  });
  const rounds = Array.from(roundMap.values());

  return (
    <div className="space-y-6 max-w-5xl mx-auto w-full">
      {/* Pinned disclaimer — always visible at top */}
      <div className="rounded-xl border border-amber-700/60 bg-amber-950/30 px-4 py-3 flex items-center gap-3 w-full max-w-full overflow-hidden">
        <FlaskConical className="w-4 h-4 text-amber-400 shrink-0" />
        <p className="text-xs text-amber-300 break-words min-w-0 flex-1">
          <strong>Results from free-form scenarios are illustrative only</strong> and are not
          covered by this project&apos;s N=30 statistical benchmarks; they should not be cited the
          same way as the business_deal or other benchmark table results.
        </p>
      </div>

      {/* Outcome banner */}
      <div
        className={`rounded-2xl border p-4 sm:p-6 backdrop-blur-xl w-full max-w-full overflow-hidden ${
          agreed
            ? 'border-emerald-800/60 bg-emerald-950/20'
            : 'border-rose-800/60 bg-rose-950/10'
        }`}
      >
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div className="flex items-center gap-3 min-w-0">
            <div
              className={`w-10 h-10 rounded-xl flex items-center justify-center shrink-0 ${
                agreed
                  ? 'bg-emerald-500/20 border border-emerald-500/40 text-emerald-400'
                  : 'bg-rose-500/20 border border-rose-500/40 text-rose-400'
              }`}
            >
              {agreed ? (
                <CheckCircle2 className="w-6 h-6" />
              ) : (
                <XCircle className="w-6 h-6" />
              )}
            </div>
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-lg font-bold text-zinc-100">
                  {agreed ? 'Agreement Reached' : 'No Agreement'}
                </span>
                <span className="text-xs font-mono px-2 py-0.5 rounded-full bg-zinc-800 text-zinc-300 border border-zinc-700/50">
                  {outcome?.rounds_taken} rounds
                </span>
              </div>
              <p className="text-xs text-zinc-400">Free-form scenario · Exploratory run</p>
            </div>
          </div>

          {outcome?.per_agent_utilities && (
            <div className="flex flex-wrap items-center gap-2 sm:gap-3 bg-zinc-950/80 border border-zinc-800/80 p-3 rounded-xl text-xs font-mono max-w-full">
              {Object.entries(outcome.per_agent_utilities).map(([agent, util]: [string, any]) => (
                <div key={agent} className="text-center px-2">
                  <p className="text-zinc-500 text-[10px] uppercase">{agent}</p>
                  <p className="text-sm font-bold text-emerald-400">{util.toFixed(3)}</p>
                </div>
              ))}
            </div>
          )}
        </div>

        {outcome?.final_proposal && (
          <div className="mt-4 pt-4 border-t border-zinc-800/40">
            <p className="text-xs font-semibold text-zinc-300 mb-2">Settlement Values:</p>
            <div className="flex flex-wrap gap-2">
              {Object.entries(outcome.final_proposal).map(([k, v]: [string, any]) => (
                <span
                  key={k}
                  className="text-xs font-mono px-2.5 py-1 rounded-lg bg-zinc-900 border border-zinc-800 text-zinc-200"
                >
                  <strong className="text-zinc-400">{k}:</strong>{' '}
                  {typeof v === 'number' ? v.toFixed(2) : v}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Round-by-round trace */}
      {rounds.length > 0 && (
        <div className="space-y-4">
          <h3 className="text-sm font-semibold text-zinc-200 flex items-center gap-2">
            <Scale className="w-4 h-4 text-zinc-400" />
            Round-by-Round Execution Trace
          </h3>
          {rounds.map((r) => (
            <div
              key={r.round}
              className="rounded-2xl border border-zinc-800/80 bg-zinc-900/40 p-5 space-y-3"
            >
              <div className="flex items-center gap-2 pb-2 border-b border-zinc-800/60">
                <span className="w-6 h-6 rounded-full bg-zinc-800 text-zinc-300 font-mono text-xs flex items-center justify-center font-bold">
                  {r.round}
                </span>
                <span className="font-semibold text-sm text-zinc-100">
                  Round {r.round} Mediation Loop
                </span>
              </div>
              {r.mediatorReasoning && (
                <p className="text-xs text-zinc-400 italic border-l-2 border-amber-500/40 pl-3">
                  &ldquo;{r.mediatorReasoning}&rdquo;
                </p>
              )}
              {r.critiques.length > 0 && (
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                  {r.critiques.map((c: any) => (
                    <div
                      key={c.agent}
                      className={`rounded-xl border p-3 text-xs space-y-1.5 ${
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
                      <p className="text-zinc-400 text-[11px] leading-relaxed line-clamp-3">
                        &ldquo;{c.reasoning}&rdquo;
                      </p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      <button
        id="scenario-builder-reset-btn"
        onClick={onReset}
        className="flex items-center gap-2 text-xs text-zinc-400 hover:text-zinc-200 border border-zinc-800 hover:border-zinc-700 px-4 py-2 rounded-lg transition-colors"
      >
        <ArrowLeft className="w-3 h-3" />
        New Scenario
      </button>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Main export
// ─────────────────────────────────────────────────────────────────────────────

type Screen = 'input' | 'confirm' | 'results';

export const ScenarioBuilder: React.FC = () => {
  const [screen, setScreen] = useState<Screen>('input');
  const [description, setDescription] = useState('');
  const [seed, setSeed] = useState(42);
  const [maxRounds, setMaxRounds] = useState(5);

  const [parsing, setParsing] = useState(false);
  const [parseError, setParseError] = useState<string | null>(null);
  const [parsed, setParsed] = useState<ParseScenarioResponse | null>(null);

  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<SessionResponse | null>(null);

  const handleParse = async () => {
    setParsing(true);
    setParseError(null);
    try {
      const res = await parseScenario({ description, seed });
      setParsed(res);
      setScreen('confirm');
    } catch (e: any) {
      let msg = e.message || 'Failed to parse scenario.';
      if (msg === 'Failed to fetch' || msg.includes('NetworkError') || msg.includes('Failed to reach')) {
        msg = `Failed to reach backend at ${API_BASE}. If using Render free tier, the backend may be waking up from cold start (takes 30–60s) — please wait a moment and retry. Also verify NEXT_PUBLIC_API_URL is configured in your Vercel project.`;
      }
      setParseError(msg);
    } finally {
      setParsing(false);
    }
  };

  const handleConfirm = async () => {
    if (!parsed) return;
    setRunning(true);
    try {
      const res = await runNegotiation({
        scenario: 'free_form',
        max_rounds: maxRounds,
        seed,
        parsed_scenario_token: parsed.parsed_scenario_token,
      });
      setResult(res);
      setScreen('results');
    } catch (e: any) {
      let msg = e.message || 'Negotiation failed.';
      if (msg === 'Failed to fetch' || msg.includes('NetworkError')) {
        msg = `Connection to ${API_BASE} lost during negotiation. If your backend is sleeping or spinning up, wait for it to wake and retry.`;
      }
      setParseError(msg);
      setScreen('confirm');
    } finally {
      setRunning(false);
    }
  };

  const handleReset = () => {
    setScreen('input');
    setParsed(null);
    setResult(null);
    setParseError(null);
  };

  return (
    <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 overflow-x-hidden">
      {screen === 'input' && (
        <InputScreen
          description={description}
          seed={seed}
          loading={parsing}
          error={parseError}
          onDescriptionChange={setDescription}
          onSeedChange={setSeed}
          onParse={handleParse}
        />
      )}
      {screen === 'confirm' && parsed && (
        <ConfirmScreen
          parsed={parsed}
          maxRounds={maxRounds}
          running={running}
          onMaxRoundsChange={setMaxRounds}
          onConfirm={handleConfirm}
          onBack={() => setScreen('input')}
        />
      )}
      {screen === 'results' && result && parsed && (
        <ResultsScreen result={result} parsed={parsed} onReset={handleReset} />
      )}
    </div>
  );
};
