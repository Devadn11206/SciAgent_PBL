export type ExperimentId = 'burgers' | 'allen-cahn' | 'nlse' | 'duffing'
export type BCKind = 'Dirichlet' | 'Neumann' | 'Periodic'
export type Invariant = 'None' | 'L2 Energy Conservation' | 'Mass Conservation'
export type IconKey = 'waves' | 'battery' | 'fiber' | 'wing'

export interface ParamDef {
  symbol: string // key, e.g. "nu"
  label: string // display glyph, e.g. "ν"
  token: string // LaTeX token used for auto-detection, e.g. "\\nu"
  description: string
  mode: 'fixed' | 'inverse'
  value: number
  min: number
  max: number
  step: number
  guess: number
  lo: number
  hi: number
  ensemble: boolean
  sensorFile: string | null
  truth?: number
}

export interface ConditionDef {
  id: string
  role: 'Initial' | 'Boundary'
  kind: BCKind
  where: string
  expr: string
}

export type LogLevel = 'INFO' | 'SYMPY' | 'AGENT' | 'OK' | 'WARN' | 'CRITIC'
export interface LogLine {
  stage: number // 0..9, see Orchestrator stage table
  progress: number // 0..1 fraction of training curves revealed
  level: LogLevel
  text: string
}

export interface FieldSpec {
  kind: 'burgers' | 'allenCahn' | 'nlse' | 'duffing'
  nx: number
  ny: number
  xRange: [number, number]
  yRange: [number, number]
  xLabel: string
  yLabel: string
  uLabel: string
  sliceVar: string // "t" or "γ"
  sliceFractions: number[]
  errGrowth: number // error growth along x (chaotic sensitivity for ODEs)
  uqRel: number // calibrated 95% UQ half-width scale, relative to max|u|
}

export interface LossSpec {
  epochs: number
  step: number
  initial: number
  interventions: { epoch: number; label: string }[]
  plateaus: number[] // PDE-residual loss plateau per phase (interventions + 1 entries)
}

export interface BenchRow {
  method: string
  detail: string
  kind: 'reference' | 'baseline' | 'agent' | 'fno'
  relL2: number | null
  linf: number | null
  meanRes: number
  driftPct: number
  wallS: number
  memMB: number
}

export interface AblationRow {
  trial: string
  action: string
  diagnosis: string
  relL2: number
}

export interface Kpis {
  relL2: string
  driftLabel: string
  drift: string
  paramLabel: string
  param: string
  runtime: string
}

export interface ReportContent {
  title: string
  subtitle: string
  keywords: string[]
  findings: string[]
  interpretation: string[]
  methods: string
  eqNote: string
}

export interface Experiment {
  id: ExperimentId
  title: string
  subtitle: string
  tint: string // pastel tailwind gradient classes
  accent: string
  icon: IconKey
  pdeName: string
  mode: 'Forward' | 'Inverse'
  isODE: boolean
  latex: string
  referenceSolver: string
  domainLabels: { x: string; t: string }
  domain: { xMin: number; xMax: number; tMin: number; tMax: number }
  uScale: number // max |u| used for normalisation
  conditions: ConditionDef[]
  invariant: Invariant
  params: ParamDef[]
  kpis: Kpis
  logs: LogLine[]
  insight: string
  agentNotes: string[] // 5 entries, one per agent node
  fieldSpec: FieldSpec
  lossSpec: LossSpec
  bench: BenchRow[]
  ablation: AblationRow[]
  report: ReportContent
}

export interface HistoryEntry {
  id: string
  expId: ExperimentId
  when: string
  status: 'verified' | 'self-corrected' | 'failed'
  relL2: string
  note: string
}


// Re-export ProblemIR and ConditionIR for convenience across views
export type { ProblemIR, ConditionIR } from '@/services/api'
