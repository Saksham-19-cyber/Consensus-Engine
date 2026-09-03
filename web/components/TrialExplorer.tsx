'use client';

import React, { useState, useEffect } from 'react';
import {
  FileText,
  Search,
  CheckCircle2,
  XCircle,
  AlertOctagon,
  ChevronDown,
  ChevronUp,
  Copy,
  Check,
  Code,
  Filter,
} from 'lucide-react';
import { listLogs, readLogRecords, LogFile, TrialRecord } from '@/lib/api';

const fallbackRecords: TrialRecord[] = [
  {
    trial_id: 'trial-001',
    scenario: 'business_deal',
    seed: 42,
    method: 'consensus_engine',
    status: 'agreed',
    rounds_taken: 2,
    agreement_reached: true,
    pareto_efficiency_ratio: 0.918,
    nash_social_welfare: 0.364,
    min_utility: 0.605,
    gini_coefficient: 0.078,
    proposal: {
      unit_price: 54.8,
      order_volume: 2150.0,
      delivery_days: 13.5,
      payment_terms: 42.0,
      quality_tier: 2.0,
    },
    utilities: { SupplierCo: 0.732, BuyerInc: 0.674, LogiTrans: 0.605 },
    bluff_suspected: false,
    messages: [
      {
        round: 1,
        agent: 'Mediator',
        content: 'Initial proposed package near range midpoints: unit_price=55.0, order_volume=2500, delivery=14d, payment=45d, quality=2.',
      },
      {
        round: 1,
        agent: 'SupplierCo',
        content: 'Price acceptable at 55.0, but would prefer slightly higher volume and shorter payment terms.',
      },
      {
        round: 1,
        agent: 'BuyerInc',
        content: 'Payment terms acceptable, price acceptable, would prefer volume closer to 2000.',
      },
      {
        round: 2,
        agent: 'Mediator',
        content: 'Revised package: unit_price=54.8, order_volume=2150, delivery=13.5d, payment=42d. Ratified by all stakeholders.',
      },
    ],
  },
  {
    trial_id: 'trial-002',
    scenario: 'business_deal',
    seed: 43,
    method: 'consensus_engine',
    status: 'agreed',
    rounds_taken: 2,
    agreement_reached: true,
    pareto_efficiency_ratio: 0.925,
    nash_social_welfare: 0.381,
    min_utility: 0.622,
    gini_coefficient: 0.065,
    proposal: {
      unit_price: 56.2,
      order_volume: 2080.0,
      delivery_days: 14.0,
      payment_terms: 40.0,
      quality_tier: 2.0,
    },
    utilities: { SupplierCo: 0.745, BuyerInc: 0.658, LogiTrans: 0.622 },
    bluff_suspected: false,
    messages: [
      { round: 1, agent: 'Mediator', content: 'Initial package proposed.' },
      { round: 2, agent: 'Mediator', content: 'Consensus ratified after volume and payment term trade-off.' },
    ],
  },
  {
    trial_id: 'trial-003',
    scenario: 'strategic_negotiation',
    seed: 101,
    method: 'consensus_engine',
    status: 'agreed',
    rounds_taken: 3,
    agreement_reached: true,
    pareto_efficiency_ratio: 0.895,
    nash_social_welfare: 0.342,
    min_utility: 0.585,
    gini_coefficient: 0.091,
    proposal: {
      unit_price: 55.4,
      order_volume: 2200.0,
      delivery_days: 13.0,
      payment_terms: 45.0,
      quality_tier: 2.0,
    },
    utilities: { SupplierCo: 0.695, BuyerInc: 0.64, LogiTrans: 0.585 },
    bluff_suspected: true,
    messages: [
      { round: 1, agent: 'SupplierCo', content: 'Stated dissatisfaction high; demand unit_price >= 85.0.' },
      {
        round: 2,
        agent: 'Mediator',
        content: 'Bluff detection triggered on SupplierCo (suppressed concessions). Holding firm on compromise terms.',
      },
      { round: 3, agent: 'SupplierCo', content: 'Concession made as deadline approaches. Deal accepted.' },
    ],
  },
  {
    trial_id: 'trial-004',
    scenario: 'roommate',
    seed: 77,
    method: 'consensus_engine',
    status: 'agreed',
    rounds_taken: 2,
    agreement_reached: true,
    pareto_efficiency_ratio: 0.941,
    nash_social_welfare: 0.412,
    min_utility: 0.641,
    gini_coefficient: 0.052,
    proposal: {
      rent_split: 51.5,
      chore_hours: 3.5,
      quiet_hours: 23.0,
      guest_nights: 2.0,
    },
    utilities: { RoommateA: 0.672, RoommateB: 0.641 },
    bluff_suspected: false,
    messages: [
      { round: 1, agent: 'Mediator', content: 'Initial chore and rent balance proposed.' },
      { round: 2, agent: 'Mediator', content: 'Both roommates agreed to 51.5% rent split with 23:00 quiet hours.' },
    ],
  },
  {
    trial_id: 'trial-005',
    scenario: 'business_deal',
    seed: 88,
    method: 'public_midpoint',
    status: 'contract_breach',
    rounds_taken: 0,
    agreement_reached: false,
    pareto_efficiency_ratio: 0.908,
    nash_social_welfare: 0.389,
    min_utility: 0.521,
    gini_coefficient: 0.072,
    proposal: {
      unit_price: 55.0,
      order_volume: 2550.0,
      delivery_days: 15.0,
      payment_terms: 45.0,
      quality_tier: 2.0,
    },
    utilities: { SupplierCo: 0.68, BuyerInc: 0.645, LogiTrans: 0.521 },
    bluff_suspected: false,
    messages: [
      {
        round: 0,
        agent: 'MidpointBaseline',
        content: 'Calculated arithmetic coordinate without checking individual rationality. Result: LogiTrans utility (0.521) breaches reservation threshold (0.650).',
      },
    ],
  },
];

export const TrialExplorer: React.FC = () => {
  const [logs, setLogs] = useState<LogFile[]>([]);
  const [selectedFile, setSelectedFile] = useState<string>('');
  const [records, setRecords] = useState<TrialRecord[]>(fallbackRecords);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [expandedTrial, setExpandedTrial] = useState<string | null>(null);
  const [jsonModalRecord, setJsonModalRecord] = useState<TrialRecord | null>(null);
  const [copied, setCopied] = useState<boolean>(false);

  useEffect(() => {
    async function loadLogList() {
      try {
        const fileList = await listLogs();
        setLogs(fileList);
        if (fileList.length > 0) {
          setSelectedFile(fileList[0].filename);
          const recs = await readLogRecords(fileList[0].filename);
          if (recs && recs.length > 0) {
            setRecords(recs);
          }
        }
      } catch (e) {
        console.warn('Unable to load logs from backend:', e);
      }
    }
    loadLogList();
  }, []);

  const handleFileChange = async (filename: string) => {
    setSelectedFile(filename);
    try {
      const recs = await readLogRecords(filename);
      setRecords(recs.length > 0 ? recs : fallbackRecords);
    } catch {
      setRecords(fallbackRecords);
    }
  };

  const filteredRecords = records.filter((r) => {
    const matchesSearch =
      !searchQuery ||
      (r.trial_id && r.trial_id.toLowerCase().includes(searchQuery.toLowerCase())) ||
      (r.scenario && r.scenario.toLowerCase().includes(searchQuery.toLowerCase())) ||
      (r.method && r.method.toLowerCase().includes(searchQuery.toLowerCase()));

    const matchesStatus =
      statusFilter === 'all' ||
      (statusFilter === 'agreed' && r.agreement_reached) ||
      (statusFilter === 'breach' && r.status === 'contract_breach') ||
      (statusFilter === 'bluff' && r.bluff_suspected);

    return matchesSearch && matchesStatus;
  });

  const handleCopy = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="space-y-8 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header Banner */}
      <div className="rounded-2xl border border-zinc-800 bg-gradient-to-b from-zinc-900/90 to-zinc-950/90 p-6 sm:p-8 backdrop-blur-xl relative overflow-hidden">
        <div className="absolute top-0 right-0 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-indigo-950/60 border border-indigo-800/40 text-indigo-400 text-xs font-mono mb-3">
              <FileText className="w-3.5 h-3.5" />
              <span>Auditable Trial Records (`data/logs/*.jsonl`)</span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-bold text-zinc-100 tracking-tight">
              Individual Trial Log Explorer
            </h1>
            <p className="text-sm text-zinc-400 mt-1 max-w-2xl">
              Inspect granular trial executions, multi-issue proposals, per-agent utility outcomes, and transcript
              critiques directly without opening raw JSONL files.
            </p>
          </div>

          {logs.length > 0 && (
            <div className="bg-zinc-900/80 border border-zinc-800 p-3 rounded-xl text-xs">
              <label className="block text-zinc-400 font-medium mb-1">Select Active JSONL File</label>
              <select
                value={selectedFile}
                onChange={(e) => handleFileChange(e.target.value)}
                className="bg-zinc-950 border border-zinc-800 rounded-lg px-2.5 py-1.5 text-zinc-200 font-mono text-[11px] focus:outline-none"
              >
                {logs.map((f) => (
                  <option key={f.filename} value={f.filename}>
                    {f.filename} ({f.size_bytes}B)
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-4 bg-zinc-900/40 p-4 rounded-xl border border-zinc-800 backdrop-blur-sm">
        <div className="relative w-full sm:w-80">
          <Search className="w-4 h-4 text-zinc-500 absolute left-3 top-2.5" />
          <input
            type="text"
            placeholder="Search by trial ID, scenario, or method..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-zinc-950 border border-zinc-800 rounded-lg pl-9 pr-3 py-1.5 text-xs text-zinc-200 focus:outline-none focus:border-indigo-500"
          />
        </div>

        <div className="flex items-center space-x-2 w-full sm:w-auto text-xs">
          <Filter className="w-3.5 h-3.5 text-zinc-400" />
          <span className="text-zinc-400">Filter:</span>
          {['all', 'agreed', 'breach', 'bluff'].map((f) => (
            <button
              key={f}
              onClick={() => setStatusFilter(f)}
              className={`px-3 py-1 rounded-lg capitalize text-xs font-medium transition-colors ${
                statusFilter === f
                  ? 'bg-indigo-500/20 text-indigo-400 border border-indigo-500/30'
                  : 'text-zinc-400 hover:text-zinc-200 bg-zinc-950'
              }`}
            >
              {f === 'breach' ? 'Breach (Midpoint)' : f === 'bluff' ? 'Bluff Detected' : f}
            </button>
          ))}
        </div>
      </div>

      {/* Trial Cards List */}
      <div className="space-y-4">
        {filteredRecords.map((r) => {
          const isExpanded = expandedTrial === r.trial_id;
          const isAgreed = r.agreement_reached;
          const isBreach = r.status === 'contract_breach';

          return (
            <div
              key={r.trial_id}
              className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-5 backdrop-blur-sm space-y-4 transition-all hover:border-zinc-700"
            >
              {/* Header row */}
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                <div className="flex items-center space-x-3">
                  <span className="font-mono text-sm font-bold text-zinc-100">{r.trial_id}</span>
                  <span className="px-2 py-0.5 rounded-full bg-zinc-800 text-zinc-300 font-mono text-[10px]">
                    {r.scenario}
                  </span>
                  <span className="px-2 py-0.5 rounded-full bg-zinc-800 text-zinc-400 font-mono text-[10px]">
                    seed={r.seed}
                  </span>
                  <span className="text-xs text-zinc-500 font-mono">[{r.method}]</span>
                </div>

                <div className="flex items-center space-x-3">
                  <span
                    className={`inline-flex items-center space-x-1 px-2.5 py-0.5 rounded-full text-xs font-medium ${
                      isAgreed
                        ? 'bg-emerald-950/80 text-emerald-400 border border-emerald-800/40'
                        : isBreach
                        ? 'bg-rose-950/80 text-rose-400 border border-rose-800/40'
                        : 'bg-zinc-800 text-zinc-300'
                    }`}
                  >
                    {isAgreed ? (
                      <CheckCircle2 className="w-3.5 h-3.5" />
                    ) : (
                      <XCircle className="w-3.5 h-3.5" />
                    )}
                    <span>{r.status?.toUpperCase()}</span>
                  </span>

                  {r.bluff_suspected && (
                    <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded-full text-[10px] font-mono bg-cyan-950 text-cyan-400 border border-cyan-800/40">
                      <AlertOctagon className="w-3 h-3" />
                      <span>Bluffing Flagged</span>
                    </span>
                  )}
                </div>
              </div>

              {/* Metrics Grid */}
              <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 p-3 rounded-lg bg-zinc-950/70 border border-zinc-800/80 text-xs font-mono">
                <div>
                  <p className="text-zinc-500 text-[10px] uppercase">Pareto Ratio</p>
                  <p className="text-emerald-400 font-bold">{r.pareto_efficiency_ratio?.toFixed(3)}</p>
                </div>
                <div>
                  <p className="text-zinc-500 text-[10px] uppercase">Nash Welfare</p>
                  <p className="text-teal-400 font-bold">{r.nash_social_welfare?.toFixed(3)}</p>
                </div>
                <div>
                  <p className="text-zinc-500 text-[10px] uppercase">Min Utility</p>
                  <p className={isBreach ? 'text-rose-400 font-bold' : 'text-zinc-200'}>
                    {r.min_utility?.toFixed(3)}
                  </p>
                </div>
                <div>
                  <p className="text-zinc-500 text-[10px] uppercase">Gini Coeff</p>
                  <p className="text-zinc-300">{r.gini_coefficient?.toFixed(3)}</p>
                </div>
                <div>
                  <p className="text-zinc-500 text-[10px] uppercase">Rounds Taken</p>
                  <p className="text-zinc-300">{r.rounds_taken}</p>
                </div>
              </div>

              {/* Proposal values */}
              {r.proposal && (
                <div className="flex flex-wrap items-center gap-2 text-xs">
                  <span className="text-zinc-400 font-medium">Outcome Package:</span>
                  {Object.entries(r.proposal).map(([k, v]) => (
                    <span
                      key={k}
                      className="font-mono text-[11px] px-2 py-0.5 rounded bg-zinc-900 border border-zinc-800 text-zinc-300"
                    >
                      {k}: <strong className="text-zinc-100">{typeof v === 'number' ? v.toFixed(1) : v}</strong>
                    </span>
                  ))}
                </div>
              )}

              {/* Utilities */}
              {r.utilities && (
                <div className="flex flex-wrap items-center gap-2 text-xs">
                  <span className="text-zinc-400 font-medium">Agent Utilities:</span>
                  {Object.entries(r.utilities).map(([k, v]) => (
                    <span
                      key={k}
                      className="font-mono text-[11px] px-2 py-0.5 rounded bg-zinc-900 border border-zinc-800 text-zinc-300"
                    >
                      {k}: <strong className="text-emerald-400">{v.toFixed(3)}</strong>
                    </span>
                  ))}
                </div>
              )}

              {/* Action buttons */}
              <div className="flex items-center justify-between pt-2 border-t border-zinc-800 text-xs">
                <button
                  onClick={() => setExpandedTrial(isExpanded ? null : (r.trial_id || ''))}
                  className="flex items-center space-x-1 text-zinc-400 hover:text-zinc-100 transition-colors"
                >
                  {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                  <span>{isExpanded ? 'Hide Messages' : `View Dialogue Messages (${r.messages?.length || 0})`}</span>
                </button>

                <button
                  onClick={() => setJsonModalRecord(r)}
                  className="flex items-center space-x-1 text-indigo-400 hover:text-indigo-300 transition-colors"
                >
                  <Code className="w-3.5 h-3.5" />
                  <span>Inspect JSON</span>
                </button>
              </div>

              {/* Expanded messages */}
              {isExpanded && r.messages && (
                <div className="mt-3 space-y-2 border-t border-zinc-800/80 pt-3">
                  {r.messages.map((m, idx) => (
                    <div key={idx} className="p-3 rounded-lg bg-zinc-950 border border-zinc-800/80 text-xs space-y-1">
                      <div className="flex items-center justify-between text-[11px]">
                        <span className="font-bold text-zinc-300 font-mono">
                          Round {m.round} &middot; {m.agent}
                        </span>
                      </div>
                      <p className="text-zinc-400 italic leading-relaxed">&ldquo;{m.content}&rdquo;</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* JSON Viewer Modal */}
      {jsonModalRecord && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
          <div className="bg-zinc-950 border border-zinc-800 rounded-2xl w-full max-w-3xl max-h-[85vh] flex flex-col shadow-2xl overflow-hidden">
            <div className="flex items-center justify-between p-4 border-b border-zinc-800">
              <div className="flex items-center space-x-2">
                <FileText className="w-4 h-4 text-indigo-400" />
                <span className="font-mono text-sm font-semibold text-zinc-200">
                  {jsonModalRecord.trial_id}.json
                </span>
              </div>
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => handleCopy(JSON.stringify(jsonModalRecord, null, 2))}
                  className="flex items-center space-x-1 text-xs px-2.5 py-1 rounded bg-zinc-900 border border-zinc-800 text-zinc-300 hover:text-zinc-100"
                >
                  {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                  <span>{copied ? 'Copied' : 'Copy'}</span>
                </button>
                <button
                  onClick={() => setJsonModalRecord(null)}
                  className="text-zinc-400 hover:text-zinc-100 text-sm px-2 py-1"
                >
                  ✕
                </button>
              </div>
            </div>

            <pre className="p-4 overflow-auto text-xs font-mono text-zinc-300 bg-zinc-950/80 leading-relaxed flex-1">
              {JSON.stringify(jsonModalRecord, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
};
