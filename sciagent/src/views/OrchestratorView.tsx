import { useEffect, useMemo, useRef, useState } from 'react'
import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { BarChart3, Bot, CheckCircle2, ClipboardCheck, FileText, Gauge, Pause, Play, RefreshCcw, Search, SkipForward, Sparkles, Terminal, Timer, Variable, Workflow, Cpu, Zap } from 'lucide-react'
import type { Experiment, LogLevel } from '@/data/types'
import { buildLosses } from '@/data/fieldGenerators'
import { Badge, Card, cx } from '@/components/ui'

const NODES = [
  { name: 'Problem Analysis Agent', short: 'Problem\nAnalysis', icon: Search },
  { name: 'Method Selection Agent', short: 'Method\nSelection', icon: Workflow },
  { name: 'Execution Engine', short: 'Execution\nEngine', icon: Cpu },
  { name: 'Critic & Optimizer Agent', short: 'Critic &\nOptimizer', icon: ClipboardCheck },
  { name: 'Research Report Agent', short: 'Research\nReport', icon: FileText },
]

type NodeState = { label: string; kind: 'pending' | 'running' | 'done' | 'correcting' }

function nodeStates(stage: number): NodeState[] {
  const pending: NodeState = { label: 'Pending', kind: 'pending' }
  const running: NodeState = { label: 'Running', kind: 'running' }
  const done: NodeState = { label: 'Completed', kind: 'done' }
  const s: NodeState[] = [pending, pending, pending, pending, pending]
  // 0 analysis · 1 method · 2 exec T1 · 3 critic · 4 exec T2 · 5 critic · 6 exec T3 · 7 critic verify · 8 report · 9 done
  s[0] = stage === 0 ? running : done
  s[1] = stage < 1 ? pending : stage === 1 ? running : done
  s[2] = stage < 2 ? pending : stage === 2 ? running : stage === 4 ? { label: 'Self-Correcting Trial 2/3', kind: 'correcting' } : stage === 6 ? { label: 'Self-Correcting Trial 3/3', kind: 'correcting' } : stage >= 7 ? done : { label: 'Trial done', kind: 'done' }
  s[3] = stage < 3 ? pending : stage === 3 || stage === 5 || stage === 7 ? running : stage === 4 || stage === 6 ? { label: 'Awaiting trial', kind: 'pending' } : done
  s[4] = stage < 8 ? pending : stage === 8 ? running : done
  return s
}

const LEVEL_COLOR: Record<LogLevel, string> = {
  INFO: '#A5B4FC', SYMPY: '#67E8F9', AGENT: '#C4B5FD', OK: '#6EE7B7', WARN: '#FCD34D', CRITIC: '#F9A8D4',
}

interface Props { exp: Experiment; runKey: number; finished: Set<string> }

export function OrchestratorView({ exp, runKey, finished }: Props) {
  const key = `${exp.id}:${runKey}`
  const logs = exp.logs
  const [count, setCount] = useState(() => (finished.has(key) ? logs.length : 0))
  const [playing, setPlaying] = useState(() => !finished.has(key))
  const [selected, setSelected] = useState(2)
  const scroller = useRef<HTMLDivElement>(null)

  // restart when the experiment or run changes
  useEffect(() => {
    if (finished.has(key)) { setCount(logs.length); setPlaying(false) } else { setCount(0); setPlaying(true) }
  }, [key, logs.length, finished])

  useEffect(() => {
    if (!playing) return
    const id = setInterval(() => {
      setCount((c) => {
        if (c + 1 >= logs.length) { finished.add(key); setPlaying(false); return logs.length }
        return c + 1
      })
    }, 650)
    return () => clearInterval(id)
  }, [playing, logs.length, key, finished])

  useEffect(() => { scroller.current?.scrollTo({ top: scroller.current.scrollHeight, behavior: 'smooth' }) }, [count])

  const last = count > 0 ? logs[count - 1] : null
  const stage = last?.stage ?? 0
  const progress = last?.progress ?? 0
  const done = stage >= 9
  const states = nodeStates(stage)
  const loops = stage === 4 || stage === 6
  const allLosses = useMemo(() => buildLosses(exp), [exp])
  const shown = useMemo(() => allLosses.filter((p) => p.epoch <= progress * exp.lossSpec.epochs), [allLosses, progress, exp])
  const [hidden, setHidden] = useState<Record<string, boolean>>({})

  const replay = () => { finished.delete(key); setCount(0); setPlaying(true) }
  const skip = () => { finished.add(key); setCount(logs.length); setPlaying(false) }

  const k = exp.kpis
  const kpis = [
    { label: 'Relative L2 Error', value: k.relL2, icon: Gauge, color: '#10B981' },
    { label: k.driftLabel, value: k.drift, icon: Zap, color: '#10B981' },
    { label: 'Discovered Param', sub: k.paramLabel, value: k.param, icon: Variable, color: '#7C3AED' },
    { label: 'Runtime', value: k.runtime, icon: Timer, color: '#06B6D4' },
  ]

  return (
    <div className="space-y-5">
      <Card
        title="LangGraph Multi-Agent Experiment Orchestrator"
        icon={<Bot size={18} className="text-turq" />}
        right={
          <div className="flex items-center gap-2">
            <Badge color={done ? '#10B981' : '#7C3AED'}>{done ? <><CheckCircle2 size={12} /> Verified convergence</> : <><span className="pulse-glow h-1.5 w-1.5 rounded-full bg-royal" /> {states[2].kind === 'correcting' ? states[2].label : 'Live'}</>}</Badge>
            <button className="btn-ghost !px-3 !py-1.5 text-xs" onClick={() => setPlaying((p) => !p)} disabled={done}>{playing ? <Pause size={13} /> : <Play size={13} />}{playing ? 'Pause' : 'Resume'}</button>
            <button className="btn-ghost !px-3 !py-1.5 text-xs" onClick={skip}><SkipForward size={13} /> Skip</button>
            <button className="btn-ghost !px-3 !py-1.5 text-xs" onClick={replay}><RefreshCcw size={13} /> Replay</button>
          </div>
        }
      >
        <Workflow5 states={states} selected={selected} onSelect={setSelected} loops={loops} stage={stage} />
        <div className="mt-2 rounded-2xl border border-cyan-100 bg-cyan-50/50 px-4 py-2.5 text-[13px] text-cyan-950">
          <b className="text-cyan-700">{NODES[selected].name}:</b> {exp.agentNotes[selected]}
        </div>
      </Card>

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        {kpis.map((m) => (
          <div key={m.label} className="glass flex items-center gap-3 !rounded-full px-5 py-3">
            <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full" style={{ background: `${m.color}1a`, color: m.color }}><m.icon size={18} /></span>
            <div className="min-w-0">
              <div className="truncate text-[11px] font-semibold uppercase tracking-wide text-slate-500">{m.label}</div>
              <div className={cx('mono text-lg font-semibold leading-tight transition', done ? 'text-indigo-950' : 'text-slate-300')} style={done ? { color: m.color } : undefined}>{done ? (m.sub ? `${m.sub.replace('Discovered ', '')} = ${m.value}` : m.value) : '···'}</div>
            </div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 gap-5 xl:grid-cols-[1.15fr_1fr]">
        <Card title="Live Solution & Multi-Term Loss Curves" icon={<BarChart3 size={17} className="text-rosepink" />} right={<span className="mono text-[11px] text-slate-400">epoch {Math.round(progress * exp.lossSpec.epochs).toLocaleString()} / {exp.lossSpec.epochs.toLocaleString()}</span>}>
          <div className="h-[340px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={shown} margin={{ left: 0, right: 14, top: 14, bottom: 0 }}>
                <CartesianGrid stroke="#EDE9FE" strokeDasharray="3 3" />
                <XAxis dataKey="epoch" type="number" domain={[0, exp.lossSpec.epochs]} tick={{ fontSize: 11, fill: '#6B7280' }} tickFormatter={(v) => `${v / 1000}k`} />
                <YAxis scale="log" domain={[1e-6, 10]} allowDataOverflow tick={{ fontSize: 10, fill: '#6B7280', fontFamily: 'JetBrains Mono' }} tickFormatter={(v) => v.toExponential(0)} width={52} ticks={[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]} />
                <Tooltip formatter={(v: number) => v.toExponential(2)} labelFormatter={(l) => `epoch ${l}`} contentStyle={{ borderRadius: 14, border: '1px solid #EDE9FE', fontSize: 12 }} />
                <Legend onClick={(e) => setHidden((h) => ({ ...h, [e.dataKey as string]: !h[e.dataKey as string] }))} wrapperStyle={{ fontSize: 12, cursor: 'pointer' }} />
                {exp.lossSpec.interventions.filter((iv) => iv.epoch <= progress * exp.lossSpec.epochs).map((iv, i) => (
                  <ReferenceLine key={iv.epoch} x={iv.epoch} stroke="#7C3AED" strokeDasharray="5 4" strokeWidth={1.6} label={{ value: iv.label, position: i % 2 ? 'insideBottomRight' : 'insideTopRight', fill: '#7C3AED', fontSize: 10, fontWeight: 700 }} />
                ))}
                <Line name="PDE residual" dataKey="pde" stroke="#EC4899" strokeWidth={2.4} dot={false} isAnimationActive={false} hide={hidden.pde} />
                <Line name="BC / IC" dataKey="bc" stroke="#C084FC" strokeWidth={1.4} strokeDasharray="4 3" dot={false} isAnimationActive={false} hide={hidden.bc} />
                <Line name="Sensor data" dataKey="data" stroke="#F59E0B" strokeWidth={1.4} dot={false} isAnimationActive={false} hide={hidden.data} />
                <Line name="Validation" dataKey="val" stroke="#06B6D4" strokeWidth={2.4} dot={false} isAnimationActive={false} hide={hidden.val} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-1 text-[11px] text-slate-400">Dashed violet markers show where the Critic Agent intervened. Click a legend entry to toggle a curve.</p>
        </Card>

        <Card title="Live Console & AI Insight" icon={<Terminal size={17} className="text-royal" />} right={<Badge color="#10B981"><span className={cx('h-1.5 w-1.5 rounded-full bg-emer', playing && 'pulse-glow')} /> ollama · llama3.1:8b</Badge>}>
          <div className="overflow-hidden rounded-2xl bg-indigoDeep shadow-inner">
            <div className="flex items-center gap-1.5 border-b border-white/10 px-4 py-2">
              <span className="h-2.5 w-2.5 rounded-full bg-[#FB7185]" /><span className="h-2.5 w-2.5 rounded-full bg-amber-300/80" /><span className="h-2.5 w-2.5 rounded-full bg-emerald-400/80" />
              <span className="mono ml-2 text-[11px] text-indigo-300">sciagent@ollama — {exp.id}.run</span>
            </div>
            <div ref={scroller} className="mono h-[268px] space-y-1 overflow-y-auto px-4 py-3 text-[11.5px] leading-relaxed">
              {logs.slice(0, count).map((l, i) => (
                <div key={i} className="flex gap-2">
                  <span className="shrink-0 text-indigo-400/70">[{String(Math.floor((i * 0.65) / 60)).padStart(2, '0')}:{((i * 0.65) % 60).toFixed(1).padStart(4, '0')}]</span>
                  <span className="w-[52px] shrink-0 font-semibold" style={{ color: LEVEL_COLOR[l.level] }}>{l.level}</span>
                  <span className="text-indigo-100">{l.text}</span>
                </div>
              ))}
              {playing && <span className="inline-block h-3.5 w-1.5 translate-y-0.5 animate-pulse bg-indigo-200" />}
              {count === 0 && !playing && <span className="text-indigo-400">press Replay to stream the agent run</span>}
            </div>
          </div>
          <div className={cx('mt-3 rounded-2xl border p-3.5 transition', done ? 'border-violet-200 bg-gradient-to-br from-violet-50 to-pink-50/60' : 'border-dashed border-violet-100 bg-white/50')}>
            <div className="mb-1 flex items-center gap-1.5 text-xs font-bold uppercase tracking-wide text-royal"><Sparkles size={13} /> AI Insight</div>
            <p className="text-[13px] leading-relaxed text-indigo-950">{done ? exp.insight : 'The Critic Agent will summarise its diagnosis once the run converges…'}</p>
          </div>
        </Card>
      </div>
    </div>
  )
}

/* ---------------------------- 5-node SVG diagram --------------------------- */

function Workflow5({ states, selected, onSelect, loops, stage }: { states: NodeState[]; selected: number; onSelect: (i: number) => void; loops: boolean; stage: number }) {
  const W = 150
  const gap = 50
  const x0 = 25
  const cx0 = (i: number) => x0 + i * (W + gap) + W / 2
  const badgeColor = (k: NodeState['kind']) => (k === 'done' ? '#10B981' : k === 'running' ? '#7C3AED' : k === 'correcting' ? '#EC4899' : '#94A3B8')
  return (
    <svg viewBox="0 0 1000 290" className="w-full" role="img" aria-label="Five-node multi-agent workflow with feedback loop from Critic to Execution Engine">
      <defs>
        <linearGradient id="nodeGrad" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#67E8F9" stopOpacity="0.95" /><stop offset="1" stopColor="#06B6D4" stopOpacity="0.85" />
        </linearGradient>
        <linearGradient id="nodeGradDim" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#CFFAFE" stopOpacity="0.8" /><stop offset="1" stopColor="#A5F3FC" stopOpacity="0.6" />
        </linearGradient>
        <filter id="glowRose" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="4" result="b" /><feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
        <filter id="softShadow" x="-20%" y="-20%" width="140%" height="150%"><feDropShadow dx="0" dy="8" stdDeviation="8" floodColor="#06B6D4" floodOpacity="0.28" /></filter>
        <marker id="arrT" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#0891B2" /></marker>
        <marker id="arrR" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#EC4899" /></marker>
      </defs>

      {/* forward connectors */}
      {NODES.slice(0, -1).map((_, i) => {
        const active = states[i].kind === 'done' && states[i + 1].kind !== 'pending'
        return <line key={i} x1={x0 + i * (W + gap) + W} y1={150} x2={x0 + (i + 1) * (W + gap) - 2} y2={150} stroke="#0891B2" strokeWidth={2.5} strokeOpacity={active ? 1 : 0.35} className={active ? 'flow-dash' : ''} markerEnd="url(#arrT)" />
      })}

      {/* rose feedback loop: Critic -> Execution Engine */}
      <path d={`M ${cx0(3)} 95 C ${cx0(3)} 12, ${cx0(2)} 12, ${cx0(2)} 91`} fill="none" stroke="#EC4899" strokeWidth={loops ? 4 : 2.5} strokeOpacity={loops || stage >= 3 ? 1 : 0.3} filter={loops ? 'url(#glowRose)' : undefined} className={loops ? 'flow-dash' : ''} strokeLinecap="round" markerEnd="url(#arrR)" />
      <text x={(cx0(2) + cx0(3)) / 2} y={30} textAnchor="middle" fontSize="12" fontWeight="700" fill="#BE185D" fontFamily="JetBrains Mono, monospace">self-correct loop · RAR · Causal · L-BFGS</text>

      {NODES.map((n, i) => {
        const st = states[i]
        const x = x0 + i * (W + gap)
        const Icon = n.icon
        const active = st.kind === 'running' || st.kind === 'correcting'
        const lines = n.short.split('\n')
        return (
          <g key={n.name} onClick={() => onSelect(i)} style={{ cursor: 'pointer' }}>
            <rect x={x} y={95} width={W} height={110} rx={26} fill={st.kind === 'pending' ? 'url(#nodeGradDim)' : 'url(#nodeGrad)'} stroke={selected === i ? '#7C3AED' : '#fff'} strokeWidth={selected === i ? 3 : 2} filter="url(#softShadow)" className={active ? 'pulse-glow' : ''} />
            <rect x={x + 6} y={100} width={W - 12} height={40} rx={20} fill="#fff" fillOpacity={0.35} />
            <foreignObject x={x + W / 2 - 14} y={108} width={28} height={28}><div style={{ color: st.kind === 'pending' ? '#0891B2' : '#fff', display: 'flex', justifyContent: 'center' }}><Icon size={24} /></div></foreignObject>
            {lines.map((l, j) => (
              <text key={j} x={x + W / 2} y={162 + j * 18} textAnchor="middle" fontSize="15" fontWeight="700" fill={st.kind === 'pending' ? '#0E7490' : '#fff'} fontFamily="Plus Jakarta Sans, sans-serif">{l}</text>
            ))}
            <text x={x + 12} y={115} fontSize="10" fontWeight="800" fill={st.kind === 'pending' ? '#0891B2' : '#fff'} fillOpacity={0.9} fontFamily="JetBrains Mono, monospace">0{i + 1}</text>
            {/* status badge */}
            <rect x={x - 4} y={222} width={W + 8} height={26} rx={13} fill="#fff" fillOpacity={0.9} stroke={badgeColor(st.kind)} strokeOpacity={0.45} />
            <circle cx={x + 14} cy={235} r={4} fill={badgeColor(st.kind)} className={active ? 'pulse-glow' : ''} />
            <text x={x + W / 2 + 6} y={239.5} textAnchor="middle" fontSize={st.label.length > 14 ? 9.5 : 11.5} fontWeight="700" fill={badgeColor(st.kind)} fontFamily="Plus Jakarta Sans, sans-serif">{st.label}</text>
          </g>
        )
      })}
    </svg>
  )
}
