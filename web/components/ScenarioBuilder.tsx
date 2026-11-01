'use client';

/**
 * ScenarioBuilder.tsx
 * ─────────────────────────────────────────────────────────────────────────────
 * Free-form natural-language scenario input for Consensus Engine.
 *
 * SCOPE NOTICE — displayed prominently in the UI:
 * Results from free-form scenarios are illustrative only and are NOT covered
 * by the N=30 statistical benchmarks documented in the README.
 *
 * Flow: Input → Parse → Confirm / Edit → Run (Live Job) → Results
 */

import React, { useState, useEffect } from 'react';
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
  Edit3,
  Save,
  Sliders,
} from 'lucide-react';
import {
  API_BASE,
  parseScenario,
  updateScenario,
  startNegotiation,
  getNegotiationJob,
  ParseScenarioResponse,
  ParsedStakeholder,
  ParsedIssue,
  SessionResponse,
  NegotiationJobStatus,
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

/** Backend not configured banner */
const BackendNotConfiguredBanner: React.FC = () => (
  <div className="flex items-start gap-3 rounded-xl border border-rose-700/60 bg-rose-950/40 px-4 py-3 w-full">
    <AlertTriangle className="w-5 h-5 text-rose-400 mt-0.5 shrink-0" />
    <div>
      <p className="text-sm font-semibold text-rose-300">Backend API URL Not Configured</p>
      <p className="text-xs text-rose-400/90 mt-1 leading-relaxed">
        <code>NEXT_PUBLIC_API_URL</code> is not set in the environment. Please configure your
        FastAPI backend URL (e.g. in <code>.env.local</code> as <code>NEXT_PUBLIC_API_URL=http://localhost:8000</code>).
      </p>
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
    {!API_BASE && <BackendNotConfiguredBanner />}
    <ExploratorBanner />

    <div className="rounded-2xl border border-zinc-800 bg-zinc-900/60 p-4 sm:p-6 backdrop-blur-xl space-y-4 w-full max-w-full overflow-hidden">
      <div className="flex items-center gap-2 mb-2">
        <FlaskConical className="w-5 h-5 text-amber-400 shrink-0" />
        <h2 className="text-lg font-semibold text-zinc-100">Describe Your Negotiation</h2>
      </div>

      <p className="text-xs text-zinc-400 leading-relaxed">
        Write a plain-language description of the negotiation situation — who the parties are,
        what they&apos;re negotiating over, and each party&apos;s priorities or constraints.
        Continuous issues (dollars, days, shares) and ordinal options (e.g. 1=basic, 2=standard, 3=dedicated)
        are supported. Pure yes/no issues should be expressed as ordinal tiers. Any party you don&apos;t fully
        specify will be labeled <span className="text-amber-300 font-mono">LLM-inferred</span> in the confirmation step.
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
          placeholder={`Example:\n\nA software startup (FounderCo) is negotiating a licensing deal with a large enterprise (EnterpriseCorp). The key issues are: annual license fee ($50,000 to $500,000), contract duration (1 to 5 years), and service tier (1=basic email support, 2=standard 24/7 support, 3=dedicated account executive and custom SLA).\n\nFounderCo strongly needs cash flow so they prefer higher fees and longer contracts with basic service tier. EnterpriseCorp has budget constraints preferring fees under $200,000, but insists on dedicated service tier (3).`}
          className="w-full max-w-full box-border bg-zinc-950/80 border border-zinc-800 rounded-xl px-4 py-3 text-sm text-zinc-200 placeholder-zinc-600 focus:outline-none focus:border-amber-500/60 focus:ring-1 focus:ring-amber-500/20 resize-y leading-relaxed font-mono whitespace-pre-wrap break-words"
        />
        <p className="text-[11px] text-zinc-600 mt-1">
          Min 20 characters. 2–6 quantifiable issues will be extracted.
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
          disabled={loading || description.trim().length < 20 || !API_BASE}
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
// State 2: Confirm & Edit Screen
// ─────────────────────────────────────────────────────────────────────────────

interface ConfirmScreenProps {
  parsed: ParseScenarioResponse;
  protocol: string;
  modelConfig: string;
  maxRounds: number;
  running: boolean;
  liveJob: NegotiationJobStatus | null;
  onProtocolChange: (v: string) => void;
  onModelConfigChange: (v: string) => void;
  onMaxRoundsChange: (v: number) => void;
  onParsedUpdate: (updated: ParseScenarioResponse) => void;
  onConfirm: () => void;
  onBack: () => void;
}

const ConfirmScreen: React.FC<ConfirmScreenProps> = ({
  parsed,
  protocol,
  modelConfig,
  maxRounds,
  running,
  liveJob,
  onProtocolChange,
  onModelConfigChange,
  onMaxRoundsChange,
  onParsedUpdate,
  onConfirm,
  onBack,
}) => {
  const [showWeights, setShowWeights] = useState<Record<string, boolean>>({});
  const [isEditing, setIsEditing] = useState(false);
  const [editedIssues, setEditedIssues] = useState<ParsedIssue[]>(parsed.issues);
  const [editedStakeholders, setEditedStakeholders] = useState<ParsedStakeholder[]>(parsed.stakeholders);
  const [savingEdits, setSavingEdits] = useState(false);
  const [editError, setEditError] = useState<string | null>(null);

  useEffect(() => {
    setEditedIssues(parsed.issues);
    setEditedStakeholders(parsed.stakeholders);
  }, [parsed]);

  const toggleWeights = (name: string) => {
    setShowWeights((prev) => ({ ...prev, [name]: !prev[name] }));
  };

  const handleIssueChange = (index: number, field: keyof ParsedIssue, value: any) => {
    setEditedIssues((prev) => {
      const copy = [...prev];
      copy[index] = { ...copy[index], [field]: value };
      return copy;
    });
  };

  const handleStakeholderReservationChange = (index: number, value: number) => {
    setEditedStakeholders((prev) => {
      const copy = [...prev];
      copy[index] = { ...copy[index], reservation_value: value, source: 'user_specified' };
      return copy;
    });
  };

  const handleStakeholderWeightChange = (index: number, issue: string, weight: number) => {
    setEditedStakeholders((prev) => {
      const copy = [...prev];
      copy[index] = {
        ...copy[index],
        weights: { ...copy[index].weights, [issue]: weight },
        source: 'user_specified',
      };
      return copy;
    });
  };

  const handleStakeholderIdealChange = (index: number, issue: string, ideal: number) => {
    setEditedStakeholders((prev) => {
      const copy = [...prev];
      copy[index] = {
        ...copy[index],
        ideal_values: { ...copy[index].ideal_values, [issue]: ideal },
        source: 'user_specified',
      };
      return copy;
    });
  };

  const handleSaveEdits = async () => {
    setSavingEdits(true);
    setEditError(null);
    try {
      const updated = await updateScenario({
        issues: editedIssues,
        stakeholders: editedStakeholders,
      });
      onParsedUpdate(updated);
      setIsEditing(false);
    } catch (e: any) {
      setEditError(e.message || 'Failed to update scenario');
    } finally {
      setSavingEdits(false);
    }
  };

  const handleLowerReservations = async () => {
    setSavingEdits(true);
    setEditError(null);
    try {
      const lowered = editedStakeholders.map((sh) => ({
        ...sh,
        reservation_value: Math.max(0.05, Math.round((sh.reservation_value - 0.10) * 100) / 100),
        source: 'user_specified',
      }));
      const updated = await updateScenario({
        issues: editedIssues,
        stakeholders: lowered,
      });
      onParsedUpdate(updated);
    } catch (e: any) {
      setEditError(e.message || 'Failed to lower reservations');
    } finally {
      setSavingEdits(false);
    }
  };

  const inferredCount = parsed.stakeholders.filter((s) => s.source === 'llm_inferred').length;

  return (
    <div className="space-y-6 max-w-5xl mx-auto w-full">
      <ExploratorBanner compact />

      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <button
          id="confirm-back-btn"
          onClick={onBack}
          disabled={running}
          className="flex items-center gap-1.5 text-xs text-zinc-400 hover:text-zinc-200 transition-colors shrink-0 disabled:opacity-50"
        >
          <ArrowLeft className="w-4 h-4" />
          Edit Description
        </button>
        <div className="flex items-center gap-2">
          {!isEditing ? (
            <button
              onClick={() => setIsEditing(true)}
              disabled={running}
              className="flex items-center gap-1.5 text-xs bg-zinc-800 hover:bg-zinc-700 text-zinc-200 px-3 py-1.5 rounded-lg border border-zinc-700 transition-colors"
            >
              <Edit3 className="w-3.5 h-3.5 text-amber-400" />
              <span>Edit Parsed Values</span>
            </button>
          ) : (
            <button
              onClick={handleSaveEdits}
              disabled={savingEdits || running}
              className="flex items-center gap-1.5 text-xs bg-emerald-600 hover:bg-emerald-500 text-zinc-950 font-semibold px-3 py-1.5 rounded-lg shadow transition-colors"
            >
              {savingEdits ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Save className="w-3.5 h-3.5" />}
              <span>Save &amp; Re-validate</span>
            </button>
          )}
          <span className="text-xs text-zinc-500">Step 2 of 3 — Confirm / Edit Scenario</span>
        </div>
      </div>

      {/* Feasibility Alert */}
      {parsed.is_feasible === false && (
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 rounded-xl border border-rose-700/60 bg-rose-950/30 p-4">
          <div className="flex items-start gap-3">
            <XCircle className="w-5 h-5 text-rose-400 mt-0.5 shrink-0" />
            <div>
              <p className="text-sm font-semibold text-rose-300">
                Infeasible Scenario: No Deal Can Satisfy Everyone As Parsed
              </p>
              <p className="text-xs text-rose-400/90 mt-1 leading-relaxed">
                There is no package within issue boundaries where every party&apos;s utility meets or
                exceeds its reservation value. Running negotiation on this setup is guaranteed to result in impasse.
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2 shrink-0 w-full sm:w-auto">
            <button
              onClick={handleLowerReservations}
              disabled={savingEdits || running}
              className="w-full sm:w-auto text-xs font-semibold px-3 py-2 rounded-lg bg-rose-600 hover:bg-rose-500 text-white transition-colors"
            >
              Lower All Reservations by 0.10
            </button>
          </div>
        </div>
      )}

      {editError && (
        <div className="flex items-start gap-2 rounded-xl border border-rose-800/50 bg-rose-950/20 px-4 py-3">
          <XCircle className="w-4 h-4 text-rose-400 mt-0.5 shrink-0" />
          <p className="text-xs text-rose-300">{editError}</p>
        </div>
      )}

      {/* LLM-inferred warning */}
      {inferredCount > 0 && (
        <div className="flex items-start gap-3 rounded-xl border border-amber-700/40 bg-amber-950/20 px-4 py-3">
          <Bot className="w-4 h-4 text-amber-400 mt-0.5 shrink-0" />
          <p className="text-xs text-amber-300 leading-relaxed">
            <strong>{inferredCount} party preference{inferredCount > 1 ? 's were' : ' was'} generated by the LLM</strong>{' '}
            because your description didn&apos;t fully specify {inferredCount > 1 ? 'their' : 'their'} utility weights,
            ideal values, or reservation value. These are plausible estimates. Review or edit them before running.
          </p>
        </div>
      )}

      {/* Issues table */}
      <div className="rounded-2xl border border-zinc-800 bg-zinc-900/60 p-5 space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-zinc-100 flex items-center gap-2">
            <Scale className="w-4 h-4 text-zinc-400" />
            Negotiation Issues ({editedIssues.length})
            {parsed.pareto_mode === 'monte_carlo' && (
              <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-amber-950/60 text-amber-300 border border-amber-700/50">
                Pareto: Monte Carlo (approximate)
              </span>
            )}
          </h3>
          {isEditing && (
            <span className="text-xs text-amber-400 font-mono">Editing Mode Active</span>
          )}
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-zinc-800">
                <th className="text-left text-zinc-500 font-medium pb-2 pr-4">Issue Name</th>
                <th className="text-left text-zinc-500 font-medium pb-2 pr-4">Min Value</th>
                <th className="text-left text-zinc-500 font-medium pb-2 pr-4">Max Value</th>
                <th className="text-left text-zinc-500 font-medium pb-2">Description / Scale Legend</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800/60">
              {editedIssues.map((issue, idx) => (
                <tr key={issue.name}>
                  <td className="py-2.5 pr-4 font-mono text-emerald-400 font-semibold">
                    {issue.name}
                  </td>
                  <td className="py-2.5 pr-4 font-mono text-zinc-300 whitespace-nowrap">
                    {isEditing ? (
                      <input
                        type="number"
                        value={issue.min_value}
                        onChange={(e) => handleIssueChange(idx, 'min_value', Number(e.target.value))}
                        className="w-20 bg-zinc-950 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-200 font-mono"
                      />
                    ) : (
                      issue.min_value
                    )}
                  </td>
                  <td className="py-2.5 pr-4 font-mono text-zinc-300 whitespace-nowrap">
                    {isEditing ? (
                      <input
                        type="number"
                        value={issue.max_value}
                        onChange={(e) => handleIssueChange(idx, 'max_value', Number(e.target.value))}
                        className="w-20 bg-zinc-950 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-200 font-mono"
                      />
                    ) : (
                      issue.max_value
                    )}
                  </td>
                  <td className="py-2.5 text-zinc-400 leading-relaxed">
                    {issue.description}
                  </td>
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
          Negotiating Parties ({editedStakeholders.length})
        </h3>
        <div className="space-y-3">
          {editedStakeholders.map((sh, sIdx) => (
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
                <div className="sm:ml-auto flex items-center gap-1.5 text-xs text-zinc-400 font-mono">
                  <span>Reservation ≥</span>
                  {isEditing ? (
                    <input
                      type="number"
                      step="0.05"
                      min="0"
                      max="1"
                      value={sh.reservation_value}
                      onChange={(e) => handleStakeholderReservationChange(sIdx, Number(e.target.value))}
                      className="w-20 bg-zinc-950 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-200 font-mono"
                    />
                  ) : (
                    <span className="text-emerald-400 font-bold">{sh.reservation_value.toFixed(2)}</span>
                  )}
                </div>
              </div>
              <p className="text-[11px] text-zinc-400 italic leading-relaxed">{sh.persona}</p>

              {/* Collapsible weights */}
              <button
                id={`toggle-weights-${sh.name}`}
                onClick={() => toggleWeights(sh.name)}
                className="flex items-center gap-1 text-[11px] text-zinc-500 hover:text-zinc-300 transition-colors"
              >
                {showWeights[sh.name] || isEditing ? (
                  <ChevronUp className="w-3 h-3" />
                ) : (
                  <ChevronDown className="w-3 h-3" />
                )}
                {showWeights[sh.name] || isEditing ? 'Hide' : 'Show'} utility weights &amp; ideal values
              </button>

              {(showWeights[sh.name] || isEditing) && (
                <div className="mt-2 overflow-x-auto">
                  <table className="text-[11px] w-full">
                    <thead>
                      <tr className="border-b border-zinc-800">
                        <th className="text-left text-zinc-600 font-medium pb-1.5 pr-3">Issue</th>
                        <th className="text-right text-zinc-600 font-medium pb-1.5 pr-3">Weight (normalized)</th>
                        <th className="text-right text-zinc-600 font-medium pb-1.5">Ideal value</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-zinc-800/40">
                      {editedIssues.map((iss) => {
                        const w = sh.weights[iss.name] ?? 0;
                        const ideal = sh.ideal_values[iss.name] ?? iss.min_value;
                        return (
                          <tr key={iss.name}>
                            <td className="py-1.5 pr-3 font-mono text-zinc-400">{iss.name}</td>
                            <td className="py-1.5 pr-3 text-right font-mono text-zinc-300">
                              {isEditing ? (
                                <input
                                  type="number"
                                  step="0.05"
                                  min="0"
                                  max="1"
                                  value={w}
                                  onChange={(e) => handleStakeholderWeightChange(sIdx, iss.name, Number(e.target.value))}
                                  className="w-20 bg-zinc-950 border border-zinc-700 rounded px-2 py-0.5 text-xs text-right font-mono text-zinc-200"
                                />
                              ) : (
                                `${(w * 100).toFixed(1)}%`
                              )}
                            </td>
                            <td className="py-1.5 text-right font-mono text-emerald-400">
                              {isEditing ? (
                                <input
                                  type="number"
                                  value={ideal}
                                  onChange={(e) => handleStakeholderIdealChange(sIdx, iss.name, Number(e.target.value))}
                                  className="w-24 bg-zinc-950 border border-zinc-700 rounded px-2 py-0.5 text-xs text-right font-mono text-emerald-400"
                                />
                              ) : (
                                ideal.toFixed(2)
                              )}
                            </td>
                          </tr>
                        );
                      })}
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

      {/* Live execution trace card (if running) */}
      {running && (
        <div className="rounded-2xl border border-amber-600/40 bg-zinc-900/80 p-5 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <RefreshCw className="w-4 h-4 text-amber-400 animate-spin" />
              <h4 className="text-sm font-semibold text-zinc-100">Live Negotiation in Progress...</h4>
            </div>
            <span className="text-xs font-mono text-zinc-400">
              {liveJob?.status === 'running' ? 'Polling updates' : 'Initializing'}
            </span>
          </div>

          {liveJob?.messages && liveJob.messages.length > 0 ? (
            <div className="space-y-2 max-h-60 overflow-y-auto pr-2">
              {liveJob.messages.map((m, idx) => (
                <div key={idx} className="text-xs bg-zinc-950/70 border border-zinc-800 rounded-lg p-2.5 space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-zinc-200">
                      R{m.round_number}: {m.agent_name} ({m.role})
                    </span>
                    <span className="text-[10px] font-mono text-zinc-500">{m.message_type}</span>
                  </div>
                  <p className="text-zinc-400 leading-relaxed italic line-clamp-2">&ldquo;{m.content}&rdquo;</p>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-xs text-zinc-500 italic">Dispatched negotiation agents to Groq LLM cluster...</p>
          )}
        </div>
      )}

      {/* Confirm & Run controls */}
      <div className="rounded-2xl border border-zinc-700/60 bg-zinc-900/80 p-5 space-y-4">
        <div className="rounded-xl border border-amber-700/50 bg-amber-950/30 px-4 py-3">
          <p className="text-xs text-amber-300 font-semibold text-center">
            ⚠ Free-form scenarios are not covered by this project&apos;s N=30 statistical
            benchmarks; results here are illustrative, not validated.
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-1">
          <div>
            <label className="block text-xs font-medium text-zinc-400 mb-1">Protocol</label>
            <select
              value={protocol}
              onChange={(e) => onProtocolChange(e.target.value)}
              disabled={running}
              className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2 text-zinc-200 text-xs focus:outline-none focus:border-emerald-500"
            >
              <option value="single_text">single_text (Mediated DAG)</option>
              <option value="alternating_offers">alternating_offers (Direct Bilateral)</option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-medium text-zinc-400 mb-1">Model Engine</label>
            <select
              value={modelConfig}
              onChange={(e) => onModelConfigChange(e.target.value)}
              disabled={running}
              className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2 text-zinc-200 text-xs focus:outline-none focus:border-emerald-500"
            >
              <option value="openai/gpt-oss-120b">openai/gpt-oss-120b</option>
              <option value="llama-3.3-70b-versatile">llama-3.3-70b-versatile</option>
              <option value="llama-3.1-8b-instant">llama-3.1-8b-instant</option>
            </select>
          </div>

          <div>
            <label className="block text-xs font-medium text-zinc-400 mb-1">Max Rounds</label>
            <input
              type="number"
              value={maxRounds}
              min={2}
              max={10}
              onChange={(e) => onMaxRoundsChange(Number(e.target.value))}
              disabled={running}
              className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2 text-zinc-200 text-xs focus:outline-none focus:border-emerald-500 font-mono"
            />
          </div>
        </div>

        <div className="flex justify-end pt-2">
          <button
            id="confirm-run-btn"
            onClick={onConfirm}
            disabled={running || parsed.is_feasible === false}
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
  const summary = result.summary;

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
        satisfaction: m.metadata?.satisfaction != null ? Number(m.metadata.satisfaction).toFixed(1) : 'n/a',
        acceptable: m.metadata?.acceptable !== undefined ? m.metadata.acceptable : 'n/a',
        concession: m.metadata?.concession_willingness != null ? `${(Number(m.metadata.concession_willingness) * 100).toFixed(0)}%` : 'n/a',
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
          same way as the fixed benchmark table results.
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
              {summary?.distance_to_frontier != null && (
                <div className="border-l border-zinc-800 pl-3 text-center">
                  <p className="text-zinc-500 text-[10px] uppercase">
                    Dist to Frontier {summary.is_approximate_frontier ? '(approx)' : ''}
                  </p>
                  <p className="text-sm font-bold text-teal-300">
                    {summary.distance_to_frontier.toFixed(3)}
                  </p>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Plain Language Package */}
        {summary?.plain_package && (
          <div className="mt-4 pt-4 border-t border-zinc-800/40 space-y-1">
            <p className="text-xs font-semibold text-zinc-300">Ratified Package in Plain Terms:</p>
            <p className="text-sm font-mono text-emerald-300 bg-zinc-950/60 p-3 rounded-xl border border-zinc-800 leading-relaxed">
              {summary.plain_package}
            </p>
          </div>
        )}

        {/* Plain English Explanation */}
        {summary?.explanation && (
          <div className="mt-4 pt-4 border-t border-zinc-800/40 space-y-1">
            <p className="text-xs font-semibold text-zinc-300">Outcome Analysis:</p>
            <p className="text-xs text-zinc-300 bg-zinc-950/40 p-3 rounded-xl border border-zinc-800/80 leading-relaxed">
              {summary.explanation}
            </p>
          </div>
        )}
      </div>

      {/* Stakeholder Utilities vs Reservations */}
      {summary?.stakeholder_outcomes && summary.stakeholder_outcomes.length > 0 && (
        <div className="rounded-2xl border border-zinc-800 bg-zinc-900/60 p-5 space-y-3">
          <h3 className="text-sm font-semibold text-zinc-100 flex items-center gap-2">
            <User className="w-4 h-4 text-zinc-400" />
            Party Satisfaction &amp; Reservation Verification
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-zinc-800">
                  <th className="text-left text-zinc-500 font-medium pb-2 pr-4">Party</th>
                  <th className="text-left text-zinc-500 font-medium pb-2 pr-4">Source</th>
                  <th className="text-right text-zinc-500 font-medium pb-2 pr-4">Outcome Utility</th>
                  <th className="text-right text-zinc-500 font-medium pb-2 pr-4">Reservation Threshold</th>
                  <th className="text-center text-zinc-500 font-medium pb-2">Individual Rationality</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-800/60">
                {summary.stakeholder_outcomes.map((so) => {
                  const shMeta = parsed.stakeholders.find((s) => s.name === so.name);
                  return (
                    <tr key={so.name}>
                      <td className="py-2.5 pr-4 font-semibold text-zinc-200">{so.name}</td>
                      <td className="py-2.5 pr-4">
                        <SourceBadge source={shMeta?.source || 'llm_inferred'} />
                      </td>
                      <td className="py-2.5 pr-4 text-right font-mono text-emerald-400 font-bold">
                        {so.utility.toFixed(3)}
                      </td>
                      <td className="py-2.5 pr-4 text-right font-mono text-zinc-400">
                        {so.reservation.toFixed(2)}
                      </td>
                      <td className="py-2.5 text-center">
                        <span
                          className={`inline-block px-2 py-0.5 rounded-full text-[10px] font-mono border ${
                            so.satisfied
                              ? 'bg-emerald-950/60 text-emerald-300 border-emerald-700/50'
                              : 'bg-rose-950/60 text-rose-300 border-rose-800/50'
                          }`}
                        >
                          {so.satisfied ? 'Satisfied (U ≥ r)' : 'Breached (U < r)'}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

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
                  Round {r.round} Loop
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
                        c.acceptable === true
                          ? 'border-emerald-900/50 bg-emerald-950/10'
                          : 'border-zinc-800 bg-zinc-950/60'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-bold text-zinc-200">{c.agent}</span>
                        <span
                          className={`text-[10px] font-mono px-2 py-0.5 rounded-full ${
                            c.acceptable === true
                              ? 'bg-emerald-900/60 text-emerald-300 border border-emerald-700/40'
                              : 'bg-rose-950/60 text-rose-300 border border-rose-800/40'
                          }`}
                        >
                          {c.acceptable === true ? 'Accept' : c.acceptable === false ? 'Critique' : 'n/a'}
                        </span>
                      </div>
                      <div className="flex justify-between text-[10px] text-zinc-400">
                        <span>Satisfaction: <strong className="text-zinc-200 font-mono">{c.satisfaction}</strong></span>
                        <span>Concession: <strong className="text-zinc-200 font-mono">{c.concession}</strong></span>
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
  const [protocol, setProtocol] = useState('single_text');
  const [modelConfig, setModelConfig] = useState('openai/gpt-oss-120b');
  const [maxRounds, setMaxRounds] = useState(5);

  const [parsing, setParsing] = useState(false);
  const [parseError, setParseError] = useState<string | null>(null);
  const [parsed, setParsed] = useState<ParseScenarioResponse | null>(null);

  const [running, setRunning] = useState(false);
  const [liveJob, setLiveJob] = useState<NegotiationJobStatus | null>(null);
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
        msg = `Failed to reach backend at ${API_BASE}. If using free tier, the backend may be waking up from cold start. Please check NEXT_PUBLIC_API_URL and try again.`;
      }
      setParseError(msg);
    } finally {
      setParsing(false);
    }
  };

  const handleConfirm = async () => {
    if (!parsed) return;
    setRunning(true);
    setLiveJob(null);
    try {
      const { job_id } = await startNegotiation({
        scenario: 'free_form',
        protocol,
        model_config: modelConfig,
        max_rounds: maxRounds,
        seed,
        parsed_scenario_token: parsed.parsed_scenario_token,
      });

      // Poll every 1s
      const maxAttempts = 120;
      let finalJob: NegotiationJobStatus | null = null;
      for (let i = 0; i < maxAttempts; i++) {
        await new Promise((r) => setTimeout(r, 1000));
        const job = await getNegotiationJob(job_id);
        setLiveJob(job);
        if (job.status === 'completed') {
          finalJob = job;
          break;
        }
        if (job.status === 'failed') {
          throw new Error(job.error || 'Negotiation failed on backend');
        }
      }

      if (!finalJob) {
        throw new Error('Negotiation timed out after 120 seconds. Please retry.');
      }

      setResult({
        session_id: job_id,
        scenario: 'free_form',
        status: 'completed',
        outcome: finalJob.outcome,
        messages: finalJob.messages,
        summary: finalJob.summary,
      });
      setScreen('results');
    } catch (e: any) {
      let msg = e.message || 'Negotiation failed.';
      if (msg === 'Failed to fetch' || msg.includes('NetworkError')) {
        msg = `Connection to ${API_BASE} lost during negotiation. Please verify backend status and retry.`;
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
    setLiveJob(null);
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
          protocol={protocol}
          modelConfig={modelConfig}
          maxRounds={maxRounds}
          running={running}
          liveJob={liveJob}
          onProtocolChange={setProtocol}
          onModelConfigChange={setModelConfig}
          onMaxRoundsChange={setMaxRounds}
          onParsedUpdate={(up) => setParsed(up)}
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
