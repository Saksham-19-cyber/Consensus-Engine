export const API_BASE = process.env.NEXT_PUBLIC_API_URL || '';

export interface NegotiateRequest {
  scenario: string;
  n_agents?: number;
  max_rounds?: number;
  seed?: number;
  protocol?: string;
  model_config?: string;
  /** Free-form mode: token returned by POST /api/scenario/parse */
  parsed_scenario_token?: string;
}

export interface StakeholderOutcome {
  name: string;
  utility: number;
  reservation: number;
  satisfied: boolean;
}

export interface CustomNegotiationSummary {
  plain_package?: string;
  stakeholder_outcomes?: StakeholderOutcome[];
  distance_to_frontier?: number | null;
  is_approximate_frontier?: boolean;
  explanation?: string;
}

export interface SessionResponse {
  session_id: string;
  scenario: string;
  status: string;
  outcome?: {
    status?: string;
    final_proposal?: Record<string, number>;
    per_agent_utilities?: Record<string, number>;
    agreement_reached?: boolean;
    rounds_taken?: number;
    protocol_used?: string;
    bluff_detection_scores?: Record<string, { avg_satisfaction: number; avg_concession: number; rounds: number }>;
  };
  messages?: Array<{
    round_number: number;
    agent_name: string;
    role: string;
    content: string;
    message_type: string;
    metadata?: Record<string, any>;
  }>;
  summary?: CustomNegotiationSummary;
}

export interface EvalReportResponse {
  scenario: string;
  summary: Record<string, {
    agreement_rate: number;
    mean_pareto_ratio: number;
    ci95_pareto_ratio: [number, number];
    mean_nash_welfare: number;
    ci95_nash_welfare: [number, number];
    mean_min_utility: number;
    mean_gini: number;
    mean_rounds: number;
    wilcoxon_pareto_vs_engine?: { p_value: number; significant: boolean } | null;
    wilcoxon_nash_vs_engine?: { p_value: number; significant: boolean } | null;
  }>;
  report_markdown: string;
}

export interface LogFile {
  filename: string;
  size_bytes: number;
  modified: number;
}

export interface TrialRecord {
  trial_id?: string;
  scenario?: string;
  seed?: number;
  method?: string;
  status?: string;
  rounds_taken?: number;
  agreement_reached?: boolean;
  pareto_efficiency_ratio?: number;
  utilitarian_welfare_ratio?: number;
  nash_social_welfare?: number;
  min_utility?: number;
  gini_coefficient?: number;
  proposal?: Record<string, number>;
  utilities?: Record<string, number>;
  near_frontier?: boolean;
  is_pareto_optimal?: boolean;
  bluff_suspected?: boolean;
  messages?: Array<{ round: number; agent: string; content: string }>;
}

// ── Free-form scenario types ──────────────────────────────────────────────────

export interface ParseScenarioRequest {
  description: string;
  seed?: number;
}

export interface ParsedIssue {
  name: string;
  min_value: number;
  max_value: number;
  description: string;
}

export interface ParsedStakeholder {
  name: string;
  role: string;
  persona: string;
  /** "user_specified" | "llm_inferred" */
  source: string;
  weights: Record<string, number>;
  ideal_values: Record<string, number>;
  reservation_value: number;
}

export interface ParseScenarioResponse {
  issues: ParsedIssue[];
  stakeholders: ParsedStakeholder[];
  field_notes: string[];
  warnings: string[];
  /** "exhaustive" | "monte_carlo" */
  pareto_mode: string;
  issue_count: number;
  is_feasible?: boolean;
  /** Opaque token — pass as parsed_scenario_token to runNegotiation() */
  parsed_scenario_token: string;
}

export interface UpdateScenarioRequest {
  issues: ParsedIssue[];
  stakeholders: ParsedStakeholder[];
}

export interface NegotiationJobStatus {
  job_id: string;
  status: 'running' | 'completed' | 'failed';
  created_at: string;
  finished_at?: string;
  error?: string | null;
  messages: Array<{
    round_number: number;
    agent_name: string;
    role: string;
    content: string;
    message_type: string;
    metadata?: Record<string, any>;
  }>;
  outcome?: any;
  summary?: CustomNegotiationSummary;
}

// ──────────────────────────────────────────────────────────────────────────────

export async function checkBackendHealth(): Promise<boolean> {
  try {
    await fetch(`${API_BASE}/docs`, { method: 'HEAD', mode: 'no-cors' });
    return true;
  } catch {
    return false;
  }
}

export async function startNegotiation(req: NegotiateRequest): Promise<{ job_id: string; status: string }> {
  const res = await fetch(`${API_BASE}/api/negotiate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `API error: ${res.status}`);
  }
  return res.json();
}

export async function getNegotiationJob(jobId: string): Promise<NegotiationJobStatus> {
  const res = await fetch(`${API_BASE}/api/negotiate/${jobId}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Job error: ${res.status}`);
  }
  return res.json();
}

export async function runNegotiation(req: NegotiateRequest): Promise<SessionResponse> {
  const { job_id } = await startNegotiation(req);
  
  // Poll until complete
  const maxAttempts = 120; // up to 120s
  for (let i = 0; i < maxAttempts; i++) {
    await new Promise((resolve) => setTimeout(resolve, 1000));
    const job = await getNegotiationJob(job_id);
    if (job.status === 'completed') {
      return {
        session_id: job_id,
        scenario: req.scenario,
        status: 'completed',
        outcome: job.outcome,
        messages: job.messages,
        summary: job.summary,
      };
    }
    if (job.status === 'failed') {
      throw new Error(job.error || 'Negotiation failed on server');
    }
  }
  throw new Error('Negotiation timed out after 120 seconds');
}

export async function parseScenario(req: ParseScenarioRequest): Promise<ParseScenarioResponse> {
  const res = await fetch(`${API_BASE}/api/scenario/parse`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Parse error: ${res.status}`);
  }
  return res.json();
}

export async function updateScenario(req: UpdateScenarioRequest): Promise<ParseScenarioResponse> {
  const res = await fetch(`${API_BASE}/api/scenario/update`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Validation error: ${res.status}`);
  }
  return res.json();
}

export async function getEvalReport(scenario: string): Promise<EvalReportResponse> {
  const res = await fetch(`${API_BASE}/api/eval/report/${scenario}`);
  if (!res.ok) {
    throw new Error(`Failed to fetch report: ${res.statusText}`);
  }
  return res.json();
}

export async function listLogs(): Promise<LogFile[]> {
  const res = await fetch(`${API_BASE}/api/logs`);
  if (!res.ok) {
    return [];
  }
  const data = await res.json();
  return data.logs || [];
}

export async function readLogRecords(filename: string): Promise<TrialRecord[]> {
  const res = await fetch(`${API_BASE}/api/logs/${filename}`);
  if (!res.ok) {
    return [];
  }
  const data = await res.json();
  return data.records || [];
}
