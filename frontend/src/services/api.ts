/**
 * SciAgent Backend API Client
 * Connects frontend to the FastAPI backend running on port 8000.
 */

export interface ParameterSpecIR {
  name: string
  symbol?: string | null
  latex_token?: string | null
  description?: string | null
  mode: 'fixed' | 'inverse'
  value?: number | null
  initial_guess?: number | null
  bounds?: [number, number] | null
}

export interface Domain1DIR {
  min_val: number
  max_val: number
}

export interface DomainSpecIR {
  t: Domain1DIR
  x?: Domain1DIR | null
  y?: Domain1DIR | null
  z?: Domain1DIR | null
}

export interface DerivativeSpecIR {
  spatial_order: number
  time_order: number
}

export interface ConditionIR {
  id: string
  role: 'Initial' | 'Boundary'
  expression: string
  location: string
  kind?: 'Dirichlet' | 'Neumann' | 'Periodic'
  order?: number
  coordinate?: 'x_min' | 'x_max' | 'periodic' | 'both'
}

export interface ProblemIR {
  id: string
  name: string
  problem_type: 'ODE' | 'PDE'
  spatial_dim?: 0 | 1 | 2 | 3
  equation_latex: string
  derivatives: DerivativeSpecIR
  domain: DomainSpecIR
  parameters: ParameterSpecIR[]
  initial_conditions: ConditionIR[]
  boundary_conditions: ConditionIR[]
  conservation_invariant: 'none' | 'l2_energy' | 'mass'
  metadata?: Record<string, any>
}

export interface ParameterDetected {
  name: string
  symbol: string
  description: string
  suggested_mode: 'fixed' | 'inverse'
  value?: number | null
  initial_guess?: number | null
  bounds?: [number, number] | null
}

export interface ParseEquationResponse {
  equation_latex: string
  typst_math: string
  problem_type: 'ODE' | 'PDE'
  spatial_dim?: number
  spatial_order: number
  time_order: number
  is_nonlinear: boolean
  regimes: string[]
  primary_regime: string
  suggested_invariant?: 'mass' | 'l2_energy' | 'none'
  stability_advisory?: string | null
  detected_parameters: ParameterDetected[]
  cfl_checks: {
    type: string
    condition: string
    max_dt_formula: string
    numerical_max_dt?: number | null
    notes?: string | null
  }[]
  scaling_factors: Record<string, number>
  well_posedness: {
    status: 'PASSED' | 'WARNING' | 'FAILED'
    is_well_posed: boolean
    messages: string[]
  }
}

export interface TimeSlice {
  fraction: number
  time: number
  values: number[]
}

export interface NumericalSolutionResponse {
  success: boolean
  method: string
  problem_id: string
  spatial_dim?: number
  x: number[]
  y?: number[] | null
  z?: number[] | null
  t: number[]
  u: number[][] // 2D grid: (Nt, Nx)
  u_2d_slices?: { t: number; grid: number[][] }[]
  slices: TimeSlice[]
  drift_pct: number
  rel_l2_error?: number | null
  linf_error?: number | null
  solve_time_sec: number
  status_message: string
}

const API_BASE = 'http://localhost:8080/api'

/**
 * Fetch 4 pre-built Canonical ProblemIR presets from backend.
 */
export async function fetchPresets(): Promise<ProblemIR[]> {
  const resp = await fetch(`${API_BASE}/presets`, {
    headers: { 'Content-Type': 'application/json' },
  })
  if (!resp.ok) {
    throw new Error(`Failed to load presets: ${resp.status} ${resp.statusText}`)
  }
  return resp.json()
}

/**
 * Parse an equation LaTeX string with SymPy on the backend.
 */
export async function parseEquation(
  equation: string,
  experimentId?: string,
): Promise<ParseEquationResponse> {
  const resp = await fetch(`${API_BASE}/problems/parse`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ equation, experiment_id: experimentId }),
  })
  if (!resp.ok) {
    const errorData = await resp.json().catch(() => ({}))
    throw new Error(errorData.detail || `Parse failed with status ${resp.status}`)
  }
  return resp.json()
}

/**
 * Execute a numerical simulation on the backend (Crank-Nicolson, SSFM, or DOP853/Radau).
 */
export async function solveProblem(
  problem: ProblemIR,
  methodOverride?: string,
  nx: number = 128,
  nt: number = 250,
): Promise<NumericalSolutionResponse> {
  const resp = await fetch(`${API_BASE}/numerical/solve`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      problem,
      method_override: methodOverride,
      nx,
      nt,
    }),
  })
  if (!resp.ok) {
    const errorData = await resp.json().catch(() => ({}))
    throw new Error(errorData.detail || `Numerical solve failed with status ${resp.status}`)
  }
  return resp.json()
}

export interface BenchmarkScore {
  method: string
  detail: string
  kind: 'reference' | 'baseline' | 'agent' | 'fno'
  rel_l2: number | null
  linf: number | null
  mean_res: number
  drift_pct: number
  wall_s: number
  mem_mb: number
}

export interface EpochMetricItem {
  epoch: number
  total_loss: number
  pde_loss: number
  bc_loss: number
  ic_loss: number
  data_loss: number
  discovered_parameters?: Record<string, number>
}

export interface TrainPINNResponse {
  success: boolean
  problem_id: string
  discovered_parameters: Record<string, number>
  parameter_uncertainties: Record<string, number>
  x: number[]
  t: number[]
  u_pred: number[][]
  u_error: number[][]
  slices: {
    fraction: number
    time: number
    values: number[]
    uq_lower?: number[]
    uq_upper?: number[]
  }[]
  rar_points: [number, number][]
  benchmark_scores: BenchmarkScore[]
  loss_history: EpochMetricItem[]
  wall_clock_sec: number
  peak_memory_mb: number
  status_message: string
}

export interface TrainPINNRequest {
  problem: ProblemIR
  adam_epochs?: number
  lbfgs_epochs?: number
  use_fourier_features?: boolean
  use_causal_loss?: boolean
  use_rar?: boolean
  ensemble_size?: number
  job_id?: string
}

/**
 * Train Nonlinear PINN with Adam + L-BFGS, RAR, Causal weights, and UQ.
 */
export async function trainPINN(req: TrainPINNRequest): Promise<TrainPINNResponse> {
  const resp = await fetch(`${API_BASE}/pinn/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  if (!resp.ok) {
    const errorData = await resp.json().catch(() => ({}))
    throw new Error(errorData.detail || `PINN training failed with status ${resp.status}`)
  }
  return resp.json()
}

export interface AgentRunResponse {
  success: boolean
  run_id: string
  problem_id: string
  status: string
  stage: number
  total_trials: number
  ablation_history: {
    trial: string
    action: string
    diagnosis: string
    relL2: number
  }[]
  diagnosis?: any
  plan?: any
  critic_verdict?: any
  research_summary?: any
  logs: {
    stage: number
    level: string
    text: string
    elapsed_s?: number
  }[]
  db_record?: any
}

export interface OllamaStatusResponse {
  online: boolean
  model: string
  supported_models: string[]
}

/**
 * Check local Ollama health status.
 */
export async function fetchOllamaStatus(): Promise<OllamaStatusResponse> {
  const resp = await fetch(`${API_BASE}/ollama/status`)
  if (!resp.ok) {
    return { online: false, model: 'offline_fallback', supported_models: [] }
  }
  return resp.json()
}

/**
 * Fetch past experiment runs from SQLite database.
 */
export async function fetchExperimentHistory(): Promise<any[]> {
  const resp = await fetch(`${API_BASE}/experiments/history`)
  if (!resp.ok) return []
  return resp.json()
}

/**
 * Fetch detailed metrics for an experiment run by ID.
 */
export async function fetchExperimentDetail(runId: string): Promise<any> {
  const resp = await fetch(`${API_BASE}/experiments/${runId}`)
  if (!resp.ok) throw new Error(`Run ${runId} not found`)
  return resp.json()
}

/**
 * Execute the 5-node cyclic LangGraph autonomous workflow on the backend.
 */
export async function runAgentWorkflow(problem: ProblemIR, runId?: string): Promise<AgentRunResponse> {
  const resp = await fetch(`${API_BASE}/agents/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ problem, max_trials: 3, run_id: runId }),
  })
  if (!resp.ok) {
    const errorData = await resp.json().catch(() => ({}))
    throw new Error(errorData.detail || `Agent workflow failed: ${resp.status}`)
  }
  return resp.json()
}

/**
 * Connect to WebSocket for streaming real-time agent node transitions.
 */
export function connectAgentWebSocket(
  runId: string,
  onTransition: (data: any) => void,
): WebSocket {
  const wsUrl = `ws://localhost:8080/ws/agents/${runId}`
  const ws = new WebSocket(wsUrl)
  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)
      onTransition(data)
    } catch {}
  }
  return ws
}
export function connectTrainWebSocket(
  jobId: string,
  onMetric: (metric: EpochMetricItem) => void,
): WebSocket {
  const wsUrl = `ws://localhost:8080/ws/train/${jobId}`
  const ws = new WebSocket(wsUrl)
  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)
      if (data.type === 'epoch_metrics') {
        onMetric(data)
      }
    } catch {
      // ignore parse error
    }
  }
  return ws
}


/**
 * Trigger publication figure, LaTeX tables, Typst PDF, and ZIP bundle generation.
 */
export async function generateReportBundle(runId: string): Promise<{
  success: boolean
  run_id: string
  pdf_path: string
  bundle_path: string
  script_path: string
  latex_tables: string
  manifest: any
}> {
  const resp = await fetch(`${API_BASE}/reports/${runId}/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  })
  if (!resp.ok) {
    throw new Error(`Report generation failed: ${resp.status}`)
  }
  return resp.json()
}

/**
 * Direct download link helpers for PDF, Bundle, and Script.
 */
export function getReportPdfUrl(runId: string): string {
  return `${API_BASE}/reports/${runId}/pdf`
}

export function getReportBundleUrl(runId: string): string {
  return `${API_BASE}/reports/${runId}/bundle`
}

export function getReportScriptUrl(runId: string): string {
  return `${API_BASE}/experiments/${runId}/export-code`
}

export function getExportCodeUrl(runId: string): string {
  return `${API_BASE}/experiments/${runId}/export-code`
}


/**
 * Upload a CSV sensor file to the backend for inverse parameter discovery.
 */
export async function uploadSensorCsv(file: File): Promise<{
  success: boolean
  filename: string
  kind: 'PDE' | 'ODE'
  row_count: number
  columns: string[]
  statistics: {
    t_min: number
    t_max: number
    u_min: number
    u_max: number
    u_mean: number
    u_std: number
    x_min?: number
    x_max?: number
  }
  preview_rows: any[]
}> {
  const formData = new FormData()
  formData.append('file', file)
  const resp = await fetch(`${API_BASE}/sensors/upload`, {
    method: 'POST',
    body: formData,
  })
  if (!resp.ok) {
    const errorData = await resp.json().catch(() => ({}))
    throw new Error(errorData.detail || `Sensor upload failed: ${resp.status}`)
  }
  return resp.json()
}

/**
 * Fetch a sample noisy sensor CSV for a preset as text.
 */
export async function fetchSampleSensorCsv(presetId: string): Promise<string> {
  const resp = await fetch(`${API_BASE}/sensors/sample/${presetId}`)
  if (!resp.ok) {
    throw new Error(`Failed to load sample sensor CSV: ${resp.status}`)
  }
  return resp.text()
}

/**
 * Download sample sensor CSV directly in browser.
 */
export function getSampleSensorDownloadUrl(presetId: string): string {
  return `${API_BASE}/sensors/sample/${presetId}`
}


/**
 * Alias for fetchExperimentDetail.
 */
export const fetchExperimentRun = fetchExperimentDetail

// ---------------------------------------------------------------------------
// Advanced FastMCP API
// ---------------------------------------------------------------------------

export interface MCPToolDef {
  name: string
  description: string
  input_schema?: any
}

export interface MCPPromptDef {
  name: string
  description: string
}

export interface MCPResourceDef {
  uri: string
  name: string
  description?: string
}

export interface MCPTraceSpanItem {
  span_id: string
  trace_id: string
  parent_span_id?: string | null
  tool_name: string
  run_id: string
  artifact_uri?: string | null
  progress_pct: number
  duration_ms: number
  status: 'started' | 'running' | 'completed' | 'error' | 'elicitation_needed'
  details?: any
  timestamp: number
}

export interface MCPManifestResponse {
  server_name: string
  protocol_version: string
  tools: MCPToolDef[]
  prompts: MCPPromptDef[]
  resources: MCPResourceDef[]
  active_tasks: any[]
  recent_trace_spans: MCPTraceSpanItem[]
  claude_desktop_config: any
}

export interface MCPElicitationSpec {
  field: string
  question: string
  options: string[]
  default?: string
}

export async function fetchMCPManifest(): Promise<MCPManifestResponse> {
  const resp = await fetch(`${API_BASE}/mcp/manifest`)
  if (!resp.ok) {
    throw new Error(`Failed to load MCP manifest: ${resp.status}`)
  }
  return resp.json()
}

export async function resolveMCPElicitation(
  runId: string,
  chosenOption: string,
  customValue?: any
): Promise<{ success: boolean; run_id: string; chosen_option: string }> {
  const resp = await fetch(`${API_BASE}/mcp/elicit/${runId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ chosen_option: chosenOption, custom_value: customValue }),
  })
  if (!resp.ok) {
    throw new Error(`Failed to resolve elicitation: ${resp.status}`)
  }
  return resp.json()
}

export async function fetchRunArtifacts(runId: string): Promise<any[]> {
  const resp = await fetch(`${API_BASE}/mcp/artifacts/${runId}`)
  if (!resp.ok) return []
  return resp.json()
}

export async function fetchRunTraces(runId: string): Promise<MCPTraceSpanItem[]> {
  const resp = await fetch(`${API_BASE}/mcp/traces/${runId}`)
  if (!resp.ok) return []
  return resp.json()
}

export async function generateProblemFromNL(description: string): Promise<ProblemIR> {
  const resp = await fetch(`${API_BASE}/problems/from-nl`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ description }),
  })
  if (!resp.ok) {
    throw new Error(`Failed to generate equation from description: ${resp.status}`)
  }
  return resp.json()
}

export async function askCriticChat(
  runId: string,
  message: string,
  history: { role: string; content: string }[] = []
): Promise<{ success: boolean; reply: string; model?: string }> {
  const resp = await fetch(`${API_BASE}/experiments/${runId}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, history }),
  })
  if (!resp.ok) {
    throw new Error(`Critic chat failed: ${resp.status}`)
  }
  return resp.json()
}

export function getSampleSensorCsvUrl(presetId: string, nProbes: number = 80, noisePct: number = 1.5): string {
  return `${API_BASE}/sensors/sample/${presetId}?n_probes=${nProbes}&noise_pct=${noisePct}`
}

export interface RecompileReportParams {
  title?: string
  authors?: string
  affiliation?: string
  journal?: string
  custom_notes?: string
  typst_override?: string
  abstract?: string
  introduction?: string
  mathematical_formulation?: string
  methodology?: string
  results_and_discussion?: string
  conclusion?: string
}

export interface RecompileReportResponse {
  success: boolean
  run_id: string
  pdf_path: string
  bundle_path?: string
  script_path?: string
  typst_source: string
  latex_tables: string
  manifest?: Record<string, any>
}

export async function recompileReport(
  runId: string,
  params: RecompileReportParams
): Promise<RecompileReportResponse> {
  const resp = await fetch(`${API_BASE}/reports/${runId}/recompile`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  })
  if (!resp.ok) {
    throw new Error(`Failed to recompile report: ${resp.status} ${resp.statusText}`)
  }
  return resp.json()
}

