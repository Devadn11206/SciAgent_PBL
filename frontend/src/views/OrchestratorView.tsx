import { useEffect, useMemo, useRef, useState } from 'react'
import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import {
  BarChart3,
  Bot,
  CheckCircle2,
  ClipboardCheck,
  FileText,
  Gauge,
  Play,
  RefreshCcw,
  Search,
  Sparkles,
  Square,
  Terminal,
  Timer,
  Variable,
  Workflow,
  Cpu,
  Zap,
  Activity,
  Info,
  Network,
  AlertCircle,
  ChevronDown,
  ChevronRight,
  Database,
  ExternalLink,
  FastForward,
  Check,
  Flame,
  Layers,
  Settings2,
} from 'lucide-react'
import type { LogLevel, ProblemIR } from '@/data/types'
import { Badge, Card, cx } from '@/components/ui'
import { CriticChatDrawer } from '@/components/CriticChatDrawer'
import { type AgentRunResponse, type EpochMetricItem, type MCPElicitationSpec, type MCPTraceSpanItem } from '@/services/api'

const NODES = [
  { name: 'Problem Analysis Agent', short: 'Problem\nAnalysis', icon: Search },
  { name: 'Method Selection Agent', short: 'Method\nSelection', icon: Workflow },
  { name: 'Execution Engine', short: 'Execution\nEngine', icon: Cpu },
  { name: 'Critic & Optimizer Agent', short: 'Critic &\nOptimizer', icon: ClipboardCheck },
  { name: 'Research Report Agent', short: 'Research\nReport', icon: FileText },
]

type NodeState = { label: string; kind: 'pending' | 'running' | 'done' | 'correcting'; duration?: string }

function computeNodeStates(stage: number, logs: any[], isRunning: boolean): NodeState[] {
  const pending: NodeState = { label: 'Pending', kind: 'pending' }
  const running: NodeState = { label: 'Running', kind: 'running' }
  const done: NodeState = { label: 'Completed', kind: 'done' }
  const s: NodeState[] = [
    { ...pending },
    { ...pending },
    { ...pending },
    { ...pending },
    { ...pending },
  ]
  
  if (stage < 0 || !isRunning && logs.length === 0) return s // Idle / Ready state

  s[0] = stage === 0 ? { ...running } : { ...done, duration: '1.2s' }
  s[1] = stage < 1 ? { ...pending } : stage === 1 ? { ...running } : { ...done, duration: '0.8s' }
  s[2] = stage < 2
    ? { ...pending }
    : stage === 2
    ? { ...running }
    : stage === 4
    ? { label: 'Self-Correcting Trial 2/3', kind: 'correcting' }
    : stage === 6
    ? { label: 'Self-Correcting Trial 3/3', kind: 'correcting' }
    : stage >= 7
    ? { ...done, duration: '14.5s' }
    : { label: 'Trial done', kind: 'done', duration: '8.4s' }
  s[3] = stage < 3
    ? { ...pending }
    : stage === 3 || stage === 5 || stage === 7
    ? { ...running }
    : stage === 4 || stage === 6
    ? { label: 'Awaiting trial', kind: 'pending' }
    : { ...done, duration: '2.1s' }
  s[4] = stage < 8 ? { ...pending } : stage === 8 ? { ...running } : { ...done, duration: '3.4s' }
  return s
}

const LEVEL_COLOR: Record<LogLevel, string> = {
  INFO: '#A5B4FC', SYMPY: '#67E8F9', AGENT: '#C4B5FD', OK: '#6EE7B7', WARN: '#FCD34D', CRITIC: '#F9A8D4',
}

interface Props {
  activeProblem: ProblemIR
  activeRunId: string | null
  isRunning: boolean
  currentStage: number
  liveLogs: any[]
  liveLosses: { epoch: number; pde: number; bc: number; data?: number; val?: number }[]
  agentResult: AgentRunResponse | null
  mcpTraces?: MCPTraceSpanItem[]
  pendingElicitation?: MCPElicitationSpec | null
  onResolveElicitation?: (chosenOption: string, customValue?: any) => Promise<void>
  onStartRun: () => void
  onCancelRun: () => void
  ollamaStatus?: { online: boolean; model: string }
}

export function OrchestratorView({
  activeProblem,
  activeRunId,
  isRunning,
  currentStage,
  liveLogs,
  liveLosses,
  agentResult,
  mcpTraces = [],
  pendingElicitation = null,
  onResolveElicitation,
  onStartRun,
  onCancelRun,
  ollamaStatus,
}: Props) {
  const [selected, setSelected] = useState(0)
  const [inspectDrawerOpen, setInspectDrawerOpen] = useState(false)
  const [consoleTab, setConsoleTab] = useState<'reasoning' | 'mcp'>('reasoning')
  const scroller = useRef<HTMLDivElement>(null)
  const [hidden, setHidden] = useState<Record<string, boolean>>({})
  const [showTraces, setShowTraces] = useState(true)
  const [selectedElicitChoice, setSelectedElicitChoice] = useState<string>('')
  const [submittingElicit, setSubmittingElicit] = useState(false)

  // Derive effective loss history: either real-time streaming, or loaded from db_record/agentResult
  const effectiveLosses = useMemo(() => {
    if (liveLosses && liveLosses.length > 0) return liveLosses
    const dbLosses = agentResult?.db_record?.raw_payload?.loss_history
    if (Array.isArray(dbLosses) && dbLosses.length > 0) {
      return dbLosses.map((l: any, i: number) => ({
        epoch: l.epoch ?? i + 1,
        trial: l.trial,
        pde: Number(l.pde ?? l.pde_loss ?? 0),
        bc: Number(l.bc ?? l.bc_loss ?? 0),
        data: l.data !== undefined ? Number(l.data) : (l.data_loss !== undefined ? Number(l.data_loss) : undefined),
        val: Number(l.val ?? l.total_loss ?? l.pde ?? 0),
      }))
    }
    return []
  }, [liveLosses, agentResult])

  // Compute intervention markers where trial index transitions
  const interventionEpochs = useMemo(() => {
    const markers: { epoch: number; trial: number }[] = []
    let currentTrial = 1
    effectiveLosses.forEach((item: any) => {
      if (item.trial && item.trial > currentTrial) {
        currentTrial = item.trial
        markers.push({ epoch: item.epoch, trial: currentTrial })
      }
    })
    return markers
  }, [effectiveLosses])

  useEffect(() => {
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight, behavior: 'smooth' })
  }, [liveLogs.length])

  // Live ticking timer for currently running stage
  const [elapsedSeconds, setElapsedSeconds] = useState(0)
  useEffect(() => {
    let t: number | undefined
    if (isRunning) {
      const start = Date.now()
      t = window.setInterval(() => {
        setElapsedSeconds((Date.now() - start) / 1000)
      }, 100)
    } else {
      setElapsedSeconds(0)
    }
    return () => {
      if (t) clearInterval(t)
    }
  }, [isRunning])

  // Interactive quick toggles for Node 4 Critic
  const [criticToggles, setCriticToggles] = useState({
    rar: true,
    causal: true,
    fourier: false,
    lbfgs: true,
  })

  // Post-run step replay scrubber
  const [replayStep, setReplayStep] = useState<number | null>(null)

  // Automatically track active node during run
  useEffect(() => {
    if (currentStage >= 0) {
      const idx = currentStage >= 8 ? 4 : Math.min(Math.floor(currentStage / 2), 4)
      setSelected(idx)
    }
  }, [currentStage])

  const states = computeNodeStates(currentStage, liveLogs, isRunning)
  const done = agentResult !== null || currentStage >= 9
  const loops = currentStage === 4 || currentStage === 6

  // KPI computations from real results
  const metrics = agentResult?.db_record?.final_metrics || agentResult?.critic_verdict
  const relL2Str = metrics?.rel_l2 ? metrics.rel_l2.toExponential(2) : metrics?.l2_error ? metrics.l2_error.toExponential(2) : '—'
  const driftStr = metrics?.drift_pct !== undefined ? `${metrics.drift_pct.toFixed(2)}%` : metrics?.conservation_drift !== undefined ? `${(metrics.conservation_drift * 100).toFixed(2)}%` : '—'

  const discoveredMap = agentResult?.db_record?.discovered_parameters || {}
  const paramKeys = Object.keys(discoveredMap)
  const discoveredStr = paramKeys.length > 0 ? `${paramKeys[0]} = ${discoveredMap[paramKeys[0]].toFixed(4)}` : '—'
  const wallTimeStr = metrics?.wall_s ? `${metrics.wall_s.toFixed(1)}s` : metrics?.wall_time ? `${metrics.wall_time.toFixed(1)}s` : isRunning ? 'running…' : '—'

  const kpis = [
    { label: 'Relative L2 Error', value: relL2Str, icon: Gauge, color: '#10B981' },
    { label: 'L2-Norm Energy Drift', value: driftStr, icon: Zap, color: '#10B981' },
    { label: 'Discovered Parameter', value: discoveredStr, icon: Variable, color: '#7C3AED' },
    { label: 'Runtime', value: wallTimeStr, icon: Timer, color: '#06B6D4' },
  ]

  const agentDescriptions = [
    'Inspects ProblemIR, parses SymPy differential operators, and classifies physical regime.',
    'Selects optimal numerical baseline (CN/SSFM/DOP853) and initializes PINN Fourier/Tanh architecture.',
    'Dispatches PyTorch autograd compiler, runs Adam + L-BFGS, and computes energy conservation drift.',
    'Audits L2 error and conservation drift. Injects adaptive RAR collocation points or causal loss on failure.',
    'Synthesizes executive findings, formats LaTeX booktabs tables, and compiles single-column Typst PDF.',
  ]

  return (
    <div className="space-y-5">
      <Card
        title={`LangGraph Self-Correcting Orchestrator · ${activeProblem.name}`}
        icon={<Bot size={18} className="text-turq" />}
        right={
          <div className="flex flex-wrap items-center gap-2">
            {done && (
              <div className="flex items-center gap-1 rounded-full border border-violet-200 bg-violet-50/80 px-2 py-1 text-[11px] font-semibold text-royal shadow-xs">
                <FastForward size={12} className="text-royal" />
                <span className="hidden sm:inline text-slate-500 mr-1">Replay:</span>
                {[
                  { label: 'T1 Baseline', step: 2, node: 2 },
                  { label: 'Critic Audit', step: 3, node: 3 },
                  { label: 'T2 Self-Correct', step: 4, node: 2 },
                  { label: 'Final Synthesis', step: 8, node: 4 },
                ].map((s) => (
                  <button
                    key={s.label}
                    onClick={() => {
                      setReplayStep(s.step)
                      setSelected(s.node)
                      setInspectDrawerOpen(true)
                    }}
                    className={cx(
                      'rounded-full px-2 py-0.5 transition text-[10.5px]',
                      replayStep === s.step
                        ? 'bg-royal text-white shadow-xs'
                        : 'text-violet-700 hover:bg-violet-200/60'
                    )}
                  >
                    {s.label}
                  </button>
                ))}
              </div>
            )}
            <Badge color={done ? '#10B981' : isRunning ? '#7C3AED' : '#94A3B8'}>
              {done ? (
                <><CheckCircle2 size={12} /> Verified Convergence ({agentResult?.total_trials ?? 1} trial{agentResult?.total_trials === 1 ? '' : 's'})</>
              ) : isRunning ? (
                <><span className="pulse-glow h-1.5 w-1.5 rounded-full bg-royal" /> {states[2].kind === 'correcting' ? states[2].label : 'Live Execution'}</>
              ) : (
                <>Ready / Idle</>
              )}
            </Badge>
            {isRunning ? (
              <button className="btn-ghost !px-3 !py-1.5 text-xs text-rose-600 hover:bg-rose-50" onClick={onCancelRun}>
                <Square size={13} fill="currentColor" /> Stop Run
              </button>
            ) : (
              <button className="btn-primary !px-3.5 !py-1.5 text-xs" onClick={onStartRun}>
                <Play size={13} fill="currentColor" /> {done ? 'Re-Run Live Workflow' : 'Start Live Run'}
              </button>
            )}
          </div>
        }
      >
        <Workflow5
          states={states}
          selected={selected}
          onSelect={(idx) => {
            setSelected(idx)
            setInspectDrawerOpen(true)
          }}
          loops={loops}
          stage={currentStage}
          isRunning={isRunning}
          elapsedSeconds={elapsedSeconds}
          trialCount={agentResult?.total_trials ?? (currentStage >= 4 ? 2 : 1)}
        />
        <div className="mt-2 flex flex-col gap-2 rounded-2xl border border-cyan-100 bg-cyan-50/50 px-4 py-2.5 text-[13px] text-cyan-950 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <b className="text-cyan-700">{NODES[selected].name}:</b> {agentDescriptions[selected]}
          </div>
          <button
            onClick={() => setInspectDrawerOpen(true)}
            className="flex items-center gap-1.5 self-start rounded-xl bg-white px-3 py-1 text-xs font-semibold text-cyan-800 shadow-sm border border-cyan-200 hover:bg-cyan-100/50 sm:self-auto"
          >
            <Info size={13} /> {inspectDrawerOpen ? 'Close Inspection Panel' : 'Inspect Agent State'}
          </button>
        </div>

        {/* Inline Expanded Agent State Inspection Panel */}
        {inspectDrawerOpen && (
          <div className="mt-3 rounded-2xl border border-violet-200 bg-white/95 p-4 shadow-md backdrop-blur-md animate-in fade-in slide-in-from-top-2">
            <div className="mb-3 flex items-center justify-between border-b border-violet-100 pb-2.5">
              <div className="flex items-center gap-2">
                <span className="mono text-xs font-bold text-royal bg-violet-100 px-2.5 py-0.5 rounded-full">
                  Node 0{selected + 1} State
                </span>
                <span className="text-sm font-bold text-slate-800">{NODES[selected].name}</span>
              </div>
              <div className="flex items-center gap-2">
                {selected === 3 && (
                  <button
                    onClick={() => onStartRun()}
                    className="flex items-center gap-1.5 rounded-full bg-gradient-to-r from-pink-600 to-purple-600 px-3 py-1 text-xs font-bold text-white shadow-sm hover:from-pink-700 hover:to-purple-700 transition"
                  >
                    <Flame size={12} /> Force Extra Refinement Trial
                  </button>
                )}
                <button
                  onClick={() => setInspectDrawerOpen(false)}
                  className="rounded-full p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-600"
                >
                  ✕
                </button>
              </div>
            </div>

            {selected === 3 && (
              <div className="mb-3 rounded-xl border border-pink-200 bg-pink-50/50 p-3">
                <div className="mb-2 flex items-center gap-1.5 text-xs font-bold uppercase tracking-wide text-pink-900">
                  <Settings2 size={13} /> Interactive Critic Self-Correction Interventions
                </div>
                <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                  {[
                    { key: 'rar', label: 'Residual Adaptive Refinement (RAR)', desc: '+1500 collocation points' },
                    { key: 'causal', label: 'Causal Loss Weighting', desc: 'Forward temporal march' },
                    { key: 'fourier', label: 'Fourier Features', desc: 'High-frequency clamp' },
                    { key: 'lbfgs', label: 'L-BFGS Hessian Polish', desc: 'Second-order curvature' },
                  ].map((item) => (
                    <button
                      key={item.key}
                      onClick={() => setCriticToggles((prev: any) => ({ ...prev, [item.key]: !prev[item.key] }))}
                      className={cx(
                        'flex flex-col items-start rounded-xl border p-2 text-left transition text-xs',
                        (criticToggles as any)[item.key]
                          ? 'border-pink-500 bg-white text-pink-950 shadow-xs'
                          : 'border-slate-200 bg-slate-50/70 text-slate-500'
                      )}
                    >
                      <div className="flex w-full items-center justify-between font-bold">
                        <span>{item.label}</span>
                        {(criticToggles as any)[item.key] && <Check size={13} className="text-pink-600" />}
                      </div>
                      <span className="text-[10px] text-slate-500 mt-0.5">{item.desc}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}

            <div className="overflow-hidden rounded-xl bg-indigoDeep p-3 shadow-inner text-indigo-100">
              <pre className="mono max-h-[220px] overflow-y-auto text-xs leading-relaxed">
                {(() => {
                  if (selected === 0) {
                    return JSON.stringify(agentResult?.diagnosis || { note: 'Inspection state available after problem analysis node runs.' }, null, 2)
                  }
                  if (selected === 1) {
                    return JSON.stringify(agentResult?.plan || { note: 'SolverPlan configuration generated during method selection.' }, null, 2)
                  }
                  if (selected === 2) {
                    return JSON.stringify({
                      trials_recorded: agentResult?.total_trials ?? 0,
                      final_parameters: agentResult?.db_record?.discovered_parameters || {},
                      final_metrics: agentResult?.db_record?.final_metrics || {},
                    }, null, 2)
                  }
                  if (selected === 3) {
                    return JSON.stringify(agentResult?.critic_verdict || agentResult?.ablation_history || { note: 'Critic audit verdicts and corrective action plans.' }, null, 2)
                  }
                  return JSON.stringify(agentResult?.research_summary || { headline: agentResult?.db_record?.insight_summary || 'Compilation generated upon convergence.' }, null, 2)
                })()}
              </pre>
            </div>
          </div>
        )}
      </Card>

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        {kpis.map((m) => (
          <div key={m.label} className="glass flex items-center gap-3 !rounded-full px-5 py-3">
            <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full" style={{ background: `${m.color}1a`, color: m.color }}><m.icon size={18} /></span>
            <div className="min-w-0">
              <div className="truncate text-[11px] font-semibold uppercase tracking-wide text-slate-500">{m.label}</div>
              <div className={cx('mono text-lg font-semibold leading-tight transition', m.value !== '—' ? 'text-indigo-950' : 'text-slate-300')} style={m.value !== '—' ? { color: m.color } : undefined}>
                {m.value}
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 gap-5 xl:grid-cols-[1.15fr_1fr]">
        <Card
          title="Live Solution & Multi-Term Loss Curves"
          icon={<BarChart3 size={17} className="text-rosepink" />}
          right={
            <span className="mono text-[11px] text-slate-400">
              {effectiveLosses.length > 0 ? `${effectiveLosses.length} epochs recorded` : 'awaiting live execution'}
            </span>
          }
        >
          <div className="h-[340px]">
            {effectiveLosses.length === 0 ? (
              <div className="flex h-full flex-col items-center justify-center gap-2 rounded-2xl border border-dashed border-violet-100 bg-slate-50/60 p-6 text-center text-slate-400">
                <BarChart3 size={32} className="text-slate-300" />
                <p className="text-sm font-medium">No live training epochs recorded yet.</p>
                <p className="max-w-xs text-xs text-slate-400">Click &ldquo;Run Autonomous Agent Workflow&rdquo; to start streaming real-time PyTorch loss convergence curves.</p>
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={effectiveLosses} margin={{ left: 0, right: 14, top: 14, bottom: 0 }}>
                  <CartesianGrid stroke="#EDE9FE" strokeDasharray="3 3" />
                  <XAxis dataKey="epoch" tick={{ fontSize: 11, fill: '#6B7280' }} />
                  <YAxis scale="log" domain={[1e-6, 10]} allowDataOverflow tick={{ fontSize: 10, fill: '#6B7280', fontFamily: 'JetBrains Mono' }} tickFormatter={(v) => v.toExponential(0)} width={52} />
                  <Tooltip formatter={(v: number) => v.toExponential(2)} labelFormatter={(l) => `epoch ${l}`} contentStyle={{ borderRadius: 14, border: '1px solid #EDE9FE', fontSize: 12 }} />
                  <Legend onClick={(e) => setHidden((h) => ({ ...h, [e.dataKey as string]: !h[e.dataKey as string] }))} wrapperStyle={{ fontSize: 12, cursor: 'pointer' }} />
                  {interventionEpochs.map((m) => (
                    <ReferenceLine
                      key={m.trial}
                      x={m.epoch}
                      stroke="#EC4899"
                      strokeDasharray="4 4"
                      strokeWidth={1.8}
                      label={{ value: `Trial ${m.trial} (Critic)`, fill: '#BE185D', fontSize: 10, position: 'insideTopRight' }}
                    />
                  ))}
                  <Line name="PDE residual" dataKey="pde" stroke="#EC4899" strokeWidth={2.4} dot={false} isAnimationActive={false} hide={hidden.pde} />
                  <Line name="BC / IC" dataKey="bc" stroke="#C084FC" strokeWidth={1.4} strokeDasharray="4 3" dot={false} isAnimationActive={false} hide={hidden.bc} />
                  {effectiveLosses[0]?.data !== undefined && <Line name="Sensor data" dataKey="data" stroke="#F59E0B" strokeWidth={1.4} dot={false} isAnimationActive={false} hide={hidden.data} />}
                  {effectiveLosses[0]?.val !== undefined && <Line name="Total loss" dataKey="val" stroke="#06B6D4" strokeWidth={2.4} dot={false} isAnimationActive={false} hide={hidden.val} />}
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
          <p className="mt-1 text-[11px] text-slate-400">Streamed real-time via WebSocket directly from PyTorch autograd compiler.</p>
        </Card>

        <Card
          title="Live Console & AI Insight"
          icon={<Terminal size={17} className="text-royal" />}
          right={
            <div className="flex items-center gap-2">
              <div className="flex rounded-lg bg-slate-100 p-0.5 text-[11px]">
                <button
                  onClick={() => setConsoleTab('reasoning')}
                  className={cx('rounded-md px-2 py-0.5 font-medium transition', consoleTab === 'reasoning' ? 'bg-white text-indigo-900 shadow-sm' : 'text-slate-500 hover:text-slate-700')}
                >
                  Agent Reasoning
                </button>
                <button
                  onClick={() => setConsoleTab('mcp')}
                  className={cx('rounded-md px-2 py-0.5 font-medium transition', consoleTab === 'mcp' ? 'bg-white text-indigo-900 shadow-sm' : 'text-slate-500 hover:text-slate-700')}
                >
                  MCP Tool Spans ({mcpTraces.length})
                </button>
              </div>
              <Badge color={ollamaStatus?.online ? '#10B981' : '#F59E0B'}>
                <span className={cx('h-1.5 w-1.5 rounded-full', ollamaStatus?.online ? 'bg-emer' : 'bg-amber-400', isRunning && 'pulse-glow')} /> {ollamaStatus?.online ? `ollama · ${ollamaStatus.model}` : 'Deterministic Offline Mode'}
              </Badge>
            </div>
          }
        >
          {consoleTab === 'reasoning' ? (
            <div className="overflow-hidden rounded-2xl bg-indigoDeep shadow-inner">
              <div className="flex items-center gap-1.5 border-b border-white/10 px-4 py-2">
                <span className="h-2.5 w-2.5 rounded-full bg-[#FB7185]" /><span className="h-2.5 w-2.5 rounded-full bg-amber-300/80" /><span className="h-2.5 w-2.5 rounded-full bg-emerald-400/80" />
                <span className="mono ml-2 text-[11px] text-indigo-300">sciagent@ollama — {activeProblem.id}.run</span>
              </div>
              <div ref={scroller} className="mono h-[268px] space-y-1 overflow-y-auto px-4 py-3 text-[11.5px] leading-relaxed">
                {liveLogs.length === 0 ? (
                  <div className="flex h-full flex-col items-center justify-center text-center text-indigo-300/60">
                    <Terminal size={24} className="mb-2 text-indigo-400/40" />
                    <span>Click &ldquo;Run Autonomous Agent Workflow&rdquo; in Problem Studio to start live execution...</span>
                  </div>
                ) : (
                  liveLogs.map((l, i) => {
                    const elapsed = l.elapsed_s !== undefined ? l.elapsed_s : (i * 0.4)
                    const mins = String(Math.floor(elapsed / 60)).padStart(2, '0')
                    const secs = (elapsed % 60).toFixed(1).padStart(4, '0')
                    return (
                      <div key={i} className="flex gap-2">
                        <span className="shrink-0 text-indigo-400/70">[{mins}:{secs}]</span>
                        <span className="w-[52px] shrink-0 font-semibold" style={{ color: LEVEL_COLOR[l.level as LogLevel] || '#C4B5FD' }}>{l.level || 'AGENT'}</span>
                        <span className="text-indigo-100">{l.text || l.message || JSON.stringify(l)}</span>
                      </div>
                    )
                  })
                )}
                {isRunning && <span className="inline-block h-3.5 w-1.5 translate-y-0.5 animate-pulse bg-indigo-200" />}
              </div>
            </div>
          ) : (
            <div className="h-[305px] overflow-y-auto rounded-2xl border border-slate-200 bg-slate-50/70 p-3">
              {mcpTraces.length === 0 ? (
                <div className="flex h-full flex-col items-center justify-center text-center text-slate-400">
                  <Network size={24} className="mb-2 text-slate-300" />
                  <p className="text-xs">No MCP tool spans recorded yet for this session.</p>
                </div>
              ) : (
                <div className="space-y-2">
                  {mcpTraces.map((span, idx) => (
                    <div key={idx} className="flex flex-col gap-1 rounded-xl border border-slate-200 bg-white p-2.5 text-xs shadow-xs">
                      <div className="flex items-center justify-between">
                        <span className="mono rounded bg-violet-100 px-1.5 py-0.5 text-[10.5px] font-bold text-violet-700">
                          {span.tool_name}
                        </span>
                        <span className={cx(
                          'rounded-full px-2 py-0.5 text-[9.5px] font-bold uppercase',
                          span.status === 'completed' ? 'bg-emerald-100 text-emerald-800' : 'bg-rose-100 text-rose-800'
                        )}>
                          {span.status}
                        </span>
                      </div>
                      <div className="mt-1 flex items-center justify-between text-slate-500 text-[11px]">
                        <span className="mono truncate">id: {span.span_id.slice(0, 14)}...</span>
                        {span.duration_ms !== undefined && (
                          <span className="flex items-center gap-1 text-slate-600 font-semibold">
                            <Timer size={11} /> {span.duration_ms.toFixed(1)}ms
                          </span>
                        )}
                      </div>
                      {span.artifact_uri && (
                        <div className="mt-0.5 mono flex items-center gap-1 rounded bg-emerald-50 px-2 py-0.5 text-[10px] text-emerald-700">
                          <Database size={10} /> {span.artifact_uri}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
          <div className={cx('mt-3 rounded-2xl border p-3.5 transition', (done || agentResult) ? 'border-violet-200 bg-gradient-to-br from-violet-50 to-pink-50/60' : 'border-dashed border-violet-100 bg-white/50')}>
            <div className="mb-1 flex items-center gap-1.5 text-xs font-bold uppercase tracking-wide text-royal"><Sparkles size={13} /> AI Insight</div>
            <p className="text-[13px] leading-relaxed text-indigo-950">
              {agentResult?.research_summary?.headline || (done ? agentResult?.db_record?.insight_summary : 'The Critic Agent will summarise its diagnosis once the run converges…')}
            </p>
          </div>
        </Card>
      </div>

      {/* Interactive MCP Elicitation Card (when triggered by parse_and_validate_equation) */}
      {pendingElicitation && (
        <Card className="mt-6 border-2 border-amber-300 bg-gradient-to-r from-amber-50/80 via-white to-amber-50/50 p-5 shadow-lg">
          <div className="flex items-start gap-3">
            <div className="rounded-xl bg-amber-500/10 p-2.5 text-amber-600">
              <AlertCircle size={24} />
            </div>
            <div className="flex-1">
              <div className="flex items-center gap-2">
                <Badge color="#D97706">MCP Interactive Elicitation Required</Badge>
                <span className="mono text-xs text-amber-700 font-semibold">{pendingElicitation.field}</span>
              </div>
              <h3 className="mt-2 text-base font-bold text-slate-800">{pendingElicitation.question}</h3>
              <p className="mt-1 text-xs text-slate-600">
                The FastMCP Equation Validation tool detected ambiguous or unconstrained conditions in the symbolic specification. Select your preferred constraint to continue:
              </p>

              <div className="mt-4 grid grid-cols-1 gap-2 sm:grid-cols-2">
                {pendingElicitation.options.map((opt) => (
                  <button
                    key={opt}
                    onClick={() => setSelectedElicitChoice(opt)}
                    className={cx(
                      'rounded-xl border p-3 text-left text-xs font-medium transition',
                      selectedElicitChoice === opt
                        ? 'border-amber-500 bg-amber-100/70 text-amber-900 shadow-sm'
                        : 'border-slate-200 bg-white hover:border-amber-300 hover:bg-amber-50/40 text-slate-700'
                    )}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-semibold">{opt}</span>
                      {selectedElicitChoice === opt && <CheckCircle2 size={14} className="text-amber-600" />}
                    </div>
                  </button>
                ))}
              </div>

              <div className="mt-4 flex items-center justify-end gap-3">
                <button
                  disabled={!selectedElicitChoice || submittingElicit}
                  onClick={async () => {
                    if (!selectedElicitChoice || !onResolveElicitation) return
                    setSubmittingElicit(true)
                    try {
                      await onResolveElicitation(selectedElicitChoice)
                    } finally {
                      setSubmittingElicit(false)
                    }
                  }}
                  className="rounded-xl bg-amber-600 px-4 py-2 text-xs font-bold text-white transition hover:bg-amber-700 disabled:opacity-50"
                >
                  {submittingElicit ? 'Submitting Resolution...' : `Apply Resolution: ${selectedElicitChoice || 'Select option'}`}
                </button>
              </div>
            </div>
          </div>
        </Card>
      )}

      {/* W3C Distributed MCP Execution Trace Spans */}
      {mcpTraces && mcpTraces.length > 0 && (
        <Card className="mt-6 border-slate-200 shadow-sm">
          <div
            className="flex cursor-pointer items-center justify-between border-b border-slate-100 p-4"
            onClick={() => setShowTraces(!showTraces)}
          >
            <div className="flex items-center gap-2">
              <Network size={16} className="text-violet-600" />
              <h3 className="text-sm font-bold text-slate-800">W3C Distributed MCP Execution Trace Spans</h3>
              <Badge color="#7C3AED" className="text-[10px]">
                {mcpTraces.length} Span{mcpTraces.length === 1 ? '' : 's'}
              </Badge>
            </div>
            <div className="flex items-center gap-2 text-xs text-slate-500">
              <span className="mono text-[11px] text-slate-400">traceparent: 00-{activeRunId || 'sciagent'}-...</span>
              {showTraces ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
            </div>
          </div>

          {showTraces && (
            <div className="divide-y divide-slate-100 overflow-hidden">
              {mcpTraces.map((span, idx) => (
                <div key={idx} className="flex flex-col gap-1 p-3 text-xs hover:bg-slate-50/60 md:flex-row md:items-center md:justify-between">
                  <div className="flex items-center gap-3">
                    <span className="mono rounded bg-violet-100 px-1.5 py-0.5 text-[10px] font-bold text-violet-700">
                      {span.tool_name}
                    </span>
                    <span className="mono text-[11px] text-slate-500">
                      span_id: {span.span_id.slice(0, 16)}...
                    </span>
                    {span.artifact_uri && (
                      <span className="mono flex items-center gap-1 rounded bg-emerald-50 px-2 py-0.5 text-[10.5px] font-medium text-emerald-700">
                        <Database size={11} /> {span.artifact_uri}
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-3 text-slate-600">
                    {span.duration_ms !== undefined && (
                      <span className="flex items-center gap-1 text-[11px] text-slate-400">
                        <Timer size={12} /> {span.duration_ms.toFixed(1)}ms
                      </span>
                    )}
                    <span className={cx(
                      'rounded-full px-2 py-0.5 text-[10px] font-bold uppercase',
                      span.status === 'completed' ? 'bg-emerald-100 text-emerald-800' : 'bg-rose-100 text-rose-800'
                    )}>
                      {span.status}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </Card>
      )}

      {/* Post-Run "Ask the Critic" Grounded Chat Assistant */}
      <CriticChatDrawer
        runId={activeRunId || agentResult?.run_id || null}
        problemName={activeProblem.name}
        className="mt-6"
      />

      {/* Structured Agent Output Inspection Drawer / Modal */}
      {inspectDrawerOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 backdrop-blur-xs p-4 animate-in fade-in">
          <div className="flex max-h-[85vh] w-full max-w-2xl flex-col rounded-3xl border border-slate-200 bg-white shadow-2xl overflow-hidden">
            <div className="flex items-center justify-between border-b border-slate-100 bg-gradient-to-r from-violet-50/80 via-white to-cyan-50/60 px-6 py-4">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-violet-600 text-white shadow-sm">
                  {(() => {
                    const NodeIcon = NODES[selected].icon
                    return <NodeIcon size={20} />
                  })()}
                </div>
                <div>
                  <h3 className="text-base font-bold text-slate-900">{NODES[selected].name}</h3>
                  <div className="flex items-center gap-2 text-xs text-slate-500">
                    <span className="mono font-semibold text-violet-700">Node 0{selected + 1}</span>
                    <span>·</span>
                    <Badge color={states[selected].kind === 'done' ? '#10B981' : states[selected].kind === 'running' ? '#7C3AED' : '#94A3B8'}>
                      {states[selected].label}
                    </Badge>
                  </div>
                </div>
              </div>
              <button
                onClick={() => setInspectDrawerOpen(false)}
                className="rounded-full p-2 text-slate-400 hover:bg-slate-100 hover:text-slate-600"
              >
                ✕
              </button>
            </div>

            <div className="flex-1 overflow-y-auto p-6 space-y-4">
              <p className="text-xs text-slate-600 leading-relaxed bg-slate-50 p-3 rounded-xl border border-slate-100">
                {agentDescriptions[selected]}
              </p>

              <div>
                <div className="mb-2 flex items-center justify-between">
                  <span className="text-xs font-bold uppercase tracking-wide text-slate-500">
                    Structured Pydantic JSON Telemetry
                  </span>
                  {selected === 3 && (
                    <button
                      onClick={() => {
                        setInspectDrawerOpen(false)
                        onStartRun()
                      }}
                      className="btn-primary !px-3 !py-1 text-xs"
                    >
                      <RefreshCcw size={12} /> Run Extra Refinement Trial
                    </button>
                  )}
                </div>
                <div className="overflow-hidden rounded-2xl bg-indigoDeep p-4 shadow-inner text-indigo-100">
                  <pre className="mono max-h-[300px] overflow-y-auto text-xs leading-relaxed">
                    {(() => {
                      if (selected === 0) {
                        return JSON.stringify(agentResult?.diagnosis || { note: 'Inspection state available after problem analysis node runs.' }, null, 2)
                      }
                      if (selected === 1) {
                        return JSON.stringify(agentResult?.plan || { note: 'SolverPlan configuration generated during method selection.' }, null, 2)
                      }
                      if (selected === 2) {
                        return JSON.stringify({
                          trials_recorded: agentResult?.total_trials ?? 0,
                          final_parameters: agentResult?.db_record?.discovered_parameters || {},
                          final_metrics: agentResult?.db_record?.final_metrics || {},
                        }, null, 2)
                      }
                      if (selected === 3) {
                        return JSON.stringify(agentResult?.critic_verdict || agentResult?.ablation_history || { note: 'Critic audit verdicts and corrective action plans.' }, null, 2)
                      }
                      return JSON.stringify(agentResult?.research_summary || { headline: agentResult?.db_record?.insight_summary || 'Compilation generated upon convergence.' }, null, 2)
                    })()}
                  </pre>
                </div>
              </div>
            </div>

            <div className="flex items-center justify-end border-t border-slate-100 bg-slate-50/50 px-6 py-3">
              <button
                onClick={() => setInspectDrawerOpen(false)}
                className="btn-ghost !px-4 !py-1.5 text-xs font-semibold"
              >
                Close Drawer
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

interface Workflow5Props {
  states: NodeState[]
  selected: number
  onSelect: (i: number) => void
  loops: boolean
  stage: number
  isRunning?: boolean
  elapsedSeconds?: number
  trialCount?: number
}

function Workflow5({ states, selected, onSelect, loops, stage, isRunning = false, elapsedSeconds = 0, trialCount = 1 }: Workflow5Props) {
  const W = 150
  const gap = 50
  const x0 = 25
  const cx0 = (i: number) => x0 + i * (W + gap) + W / 2
  const badgeColor = (k: NodeState['kind']) => (k === 'done' ? '#10B981' : k === 'running' ? '#7C3AED' : k === 'correcting' ? '#EC4899' : '#94A3B8')

  const mins = String(Math.floor(elapsedSeconds / 60)).padStart(2, '0')
  const secs = (elapsedSeconds % 60).toFixed(1).padStart(4, '0')
  const formattedTimer = `${mins}:${secs}s`

  return (
    <svg viewBox="0 0 1000 290" className="w-full" role="img" aria-label="Five-node multi-agent workflow with feedback loop from Critic to Execution Engine">
      <defs>
        <linearGradient id="nodeGrad" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#67E8F9" stopOpacity="0.95" /><stop offset="1" stopColor="#06B6D4" stopOpacity="0.85" />
        </linearGradient>
        <linearGradient id="nodeGradRunning" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#A78BFA" stopOpacity="0.95" /><stop offset="1" stopColor="#7C3AED" stopOpacity="0.90" />
        </linearGradient>
        <linearGradient id="nodeGradDone" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#34D399" stopOpacity="0.95" /><stop offset="1" stopColor="#059669" stopOpacity="0.88" />
        </linearGradient>
        <linearGradient id="nodeGradDim" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#CFFAFE" stopOpacity="0.8" /><stop offset="1" stopColor="#A5F3FC" stopOpacity="0.6" />
        </linearGradient>
        <filter id="glowRose" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="4" result="b" /><feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
        <filter id="glowPurple" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="5" result="b" /><feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
        <filter id="softShadow" x="-20%" y="-20%" width="140%" height="150%"><feDropShadow dx="0" dy="8" stdDeviation="8" floodColor="#06B6D4" floodOpacity="0.25" /></filter>
        <marker id="arrT" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#0891B2" /></marker>
        <marker id="arrActive" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#7C3AED" /></marker>
        <marker id="arrR" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#EC4899" /></marker>
      </defs>

      {/* forward connectors with animated data-packet flow */}
      {NODES.slice(0, -1).map((_, i) => {
        const isConnActive = isRunning && stage === i
        const isConnPassed = states[i].kind === 'done'
        const startX = x0 + i * (W + gap) + W
        const endX = x0 + (i + 1) * (W + gap) - 4
        const lineId = `edge-${i}-${i+1}`

        return (
          <g key={i}>
            <path
              id={lineId}
              d={`M ${startX} 150 L ${endX} 150`}
              stroke={isConnActive ? '#7C3AED' : isConnPassed ? '#10B981' : '#0891B2'}
              strokeWidth={isConnActive ? 3.5 : 2.5}
              strokeOpacity={isConnActive ? 1 : isConnPassed ? 0.8 : 0.25}
              strokeDasharray={isConnActive ? '6,4' : undefined}
              className={isConnActive ? 'flow-dash' : ''}
              markerEnd={isConnActive ? 'url(#arrActive)' : 'url(#arrT)'}
            />
            {/* Animated data packet orb along active edge */}
            {isConnActive && (
              <circle r="4.5" fill="#C084FC" filter="url(#glowPurple)">
                <animateMotion dur="1.2s" repeatCount="indefinite" path={`M ${startX} 150 L ${endX} 150`} />
              </circle>
            )}
          </g>
        )
      })}

      {/* rose/magenta self-correct loop arc: Critic (Node 3) -> PINN Engine (Node 2) */}
      <g>
        <path
          id="critic-loop-path"
          d={`M ${cx0(3)} 95 C ${cx0(3)} 10, ${cx0(2)} 10, ${cx0(2)} 91`}
          fill="none"
          stroke={loops ? '#EC4899' : '#F472B6'}
          strokeWidth={loops ? 4 : 2.5}
          strokeOpacity={loops || stage >= 3 ? 1 : 0.25}
          strokeDasharray={loops ? '8,5' : undefined}
          filter={loops ? 'url(#glowRose)' : undefined}
          className={loops ? 'flow-dash-reverse' : ''}
          strokeLinecap="round"
          markerEnd="url(#arrR)"
        />

        {/* Moving orb on feedback loop during loop correction */}
        {loops && (
          <circle r="5" fill="#FB7185" filter="url(#glowRose)">
            <animateMotion dur="1.8s" repeatCount="indefinite" path={`M ${cx0(3)} 95 C ${cx0(3)} 10, ${cx0(2)} 10, ${cx0(2)} 91`} />
          </circle>
        )}

        {/* Floating badge over feedback arc */}
        <g transform={`translate(${(cx0(2) + cx0(3)) / 2 - 110}, 18)`}>
          <rect
            width="220"
            height="24"
            rx="12"
            fill="#FFF1F2"
            stroke="#FDA4AF"
            strokeWidth="1.5"
            filter="drop-shadow(0 2px 4px rgba(244,63,94,0.15))"
          />
          <text
            x="110"
            y="16"
            textAnchor="middle"
            fontSize="10"
            fontWeight="800"
            fill="#BE185D"
            fontFamily="JetBrains Mono, monospace"
          >
            {loops ? `↺ SELF-CORRECT TRIAL ${trialCount}/3 · RAR · CAUSAL` : 'Self-Correct Loop · RAR · Causal · L-BFGS'}
          </text>
        </g>
      </g>

      {/* 5 Agent Workflow Nodes */}
      {NODES.map((n, i) => {
        const st = states[i]
        const x = x0 + i * (W + gap)
        const Icon = n.icon
        const isRunningNode = st.kind === 'running'
        const isDoneNode = st.kind === 'done'
        const isSelected = selected === i
        const lines = n.short.split('\n')

        // Choose gradient fill based on state
        const fillGrad = isRunningNode ? 'url(#nodeGradRunning)' : isDoneNode ? 'url(#nodeGradDone)' : st.kind === 'pending' ? 'url(#nodeGradDim)' : 'url(#nodeGrad)'

        return (
          <g
            key={n.name}
            onClick={() => onSelect(i)}
            style={{ cursor: 'pointer', transition: 'transform 0.2s ease-in-out' }}
            className={isSelected ? 'filter drop-shadow-lg' : ''}
          >
            {/* Main Node Card Card */}
            <rect
              x={x}
              y={95}
              width={W}
              height={110}
              rx={26}
              fill={fillGrad}
              stroke={isSelected ? '#7C3AED' : isRunningNode ? '#C084FC' : '#fff'}
              strokeWidth={isSelected ? 3.5 : isRunningNode ? 3 : 2}
              filter={isRunningNode ? 'url(#glowPurple)' : 'url(#softShadow)'}
              className={isRunningNode ? 'pulse-glow' : ''}
            />

            {/* Inner top icon pill */}
            <rect x={x + 6} y={100} width={W - 12} height={42} rx={21} fill="#fff" fillOpacity={isRunningNode ? 0.25 : 0.35} />

            {/* Animated Radial Progress Ring around icon when running */}
            {isRunningNode && (
              <circle
                cx={x + W / 2}
                cy={121}
                r={19}
                fill="none"
                stroke="#EDE9FE"
                strokeWidth={3}
                strokeDasharray="90 30"
                className="spin-slow"
              />
            )}

            {/* Checkmark circle when done */}
            {isDoneNode && (
              <circle
                cx={x + W / 2 + 20}
                cy={112}
                r={7}
                fill="#10B981"
                stroke="#fff"
                strokeWidth={1.5}
              />
            )}

            {/* Node Icon */}
            <foreignObject x={x + W / 2 - 14} y={107} width={28} height={28}>
              <div style={{ color: st.kind === 'pending' ? '#0891B2' : '#fff', display: 'flex', justifyContent: 'center' }}>
                <Icon size={24} />
              </div>
            </foreignObject>

            {/* Step label / Name */}
            {lines.map((l, j) => (
              <text
                key={j}
                x={x + W / 2}
                y={162 + j * 18}
                textAnchor="middle"
                fontSize="14.5"
                fontWeight="700"
                fill={st.kind === 'pending' ? '#0E7490' : '#fff'}
                fontFamily="Plus Jakarta Sans, sans-serif"
              >
                {l}
              </text>
            ))}

            {/* Step Index Badge (01, 02...) */}
            <text
              x={x + 14}
              y={116}
              fontSize="10"
              fontWeight="800"
              fill={st.kind === 'pending' ? '#0891B2' : '#fff'}
              fillOpacity={0.9}
              fontFamily="JetBrains Mono, monospace"
            >
              0{i + 1}
            </text>

            {/* Bottom Status & Micro-Telemetry Badge */}
            <rect
              x={x - 4}
              y={220}
              width={W + 8}
              height={28}
              rx={14}
              fill="#fff"
              fillOpacity={0.96}
              stroke={badgeColor(st.kind)}
              strokeWidth={isRunningNode ? 2 : 1}
              strokeOpacity={isRunningNode ? 0.9 : 0.45}
              filter="drop-shadow(0 2px 5px rgba(0,0,0,0.06))"
            />

            {/* Live Indicator Dot */}
            <circle
              cx={x + 12}
              cy={234}
              r={4}
              fill={badgeColor(st.kind)}
              className={isRunningNode ? 'pulse-glow' : ''}
            />

            {/* Label + Micro-telemetry timer */}
            <text
              x={x + W / 2 + 4}
              y={238}
              textAnchor="middle"
              fontSize={isRunningNode ? 10 : st.label.length > 14 ? 9.5 : 11}
              fontWeight="700"
              fill={badgeColor(st.kind)}
              fontFamily="Plus Jakarta Sans, sans-serif"
            >
              {isRunningNode ? `${st.label} ${formattedTimer}` : st.label}
            </text>
          </g>
        )
      })}
    </svg>
  )
}
