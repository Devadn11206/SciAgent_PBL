import { useMemo, useRef, useState } from 'react'
import { Battery, FileSpreadsheet, FlaskConical, Layers, Play, Plus, Radio, Ruler, ShieldCheck, Sigma, Upload, Waves, Wind, X } from 'lucide-react'
import type { BCKind, ConditionDef, Experiment, ExperimentId, Invariant, ParamDef } from '@/data/types'
import { experiments, PARAM_CATALOG } from '@/data/mockExperiments'
import { Equation } from '@/components/Equation'
import { Badge, Card, cx, Segmented, Switch } from '@/components/ui'

const ICONS = { waves: Waves, battery: Battery, fiber: Radio, wing: Wind }

const detect = (latex: string) =>
  Object.values(PARAM_CATALOG).filter((c) => (c.symbol === 'eps' ? /\\varepsilon|\\epsilon/.test(latex) : latex.includes(c.token)))

const BC_COLORS: Record<BCKind, string> = { Dirichlet: '#7C3AED', Neumann: '#06B6D4', Periodic: '#EC4899' }
const INVARIANTS: Invariant[] = ['None', 'L2 Energy Conservation', 'Mass Conservation']

interface Props {
  exp: Experiment
  onSelect: (id: ExperimentId) => void
  onRun: () => void
}

export function StudioView({ exp, onSelect, onRun }: Props) {
  return (
    <div className="space-y-5">
      <div>
        <p className="label mb-2 flex items-center gap-2"><FlaskConical size={13} /> Real-World Nonlinear Case Study — one click</p>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
          {experiments.map((e) => {
            const Icon = ICONS[e.icon]
            const active = e.id === exp.id
            return (
              <button
                key={e.id}
                onClick={() => onSelect(e.id)}
                className={cx(
                  'group relative overflow-hidden rounded-3xl border bg-gradient-to-br p-4 text-left shadow-glass backdrop-blur-md transition hover:-translate-y-0.5',
                  e.tint,
                  active ? 'border-transparent ring-2' : 'border-white',
                )}
                style={active ? ({ ['--tw-ring-color' as string]: e.accent } as React.CSSProperties) : undefined}
              >
                <div className="mb-2 flex items-center justify-between">
                  <span className="flex h-9 w-9 items-center justify-center rounded-2xl" style={{ background: `${e.accent}1f`, color: e.accent }}><Icon size={18} /></span>
                  <Badge color={e.accent}>{e.mode}</Badge>
                </div>
                <div className="text-[15px] font-bold leading-tight text-indigo-950">{e.title}</div>
                <div className="mb-2 text-xs font-medium text-violet-600">{e.subtitle}</div>
                <div className="overflow-hidden rounded-xl bg-white/70 px-2 py-1 text-[11px]"><Equation latex={e.latex} display={false} className="whitespace-nowrap [&_.katex]:text-[0.95em]" /></div>
              </button>
            )
          })}
        </div>
      </div>
      {/* keyed remount resets all editable studio state when a preset changes */}
      <StudioBody key={exp.id} exp={exp} onRun={onRun} />
    </div>
  )
}

function StudioBody({ exp, onRun }: { exp: Experiment; onRun: () => void }) {
  const [latex, setLatex] = useState(exp.latex)
  const [dom, setDom] = useState(exp.domain)
  const [conds, setConds] = useState<ConditionDef[]>(exp.conditions)
  const [inv, setInv] = useState<Invariant>(exp.invariant)
  const [norm, setNorm] = useState(true)
  const [params, setParams] = useState<Record<string, ParamDef>>(() => Object.fromEntries(exp.params.map((p) => [p.symbol, p])))

  const symbols = useMemo(() => detect(latex), [latex])
  const rows: ParamDef[] = symbols.map(
    (c) => params[c.symbol] ?? ({ ...c, mode: 'fixed', ensemble: false, sensorFile: null } as ParamDef),
  )
  const setParam = (p: ParamDef, patch: Partial<ParamDef>) => setParams((s) => ({ ...s, [p.symbol]: { ...p, ...patch } }))

  const num = (v: string, fb: number) => (Number.isFinite(parseFloat(v)) ? parseFloat(v) : fb)
  const xr = dom.xMax - dom.xMin || 1
  const tr = dom.tMax - dom.tMin || 1
  const inverseCount = rows.filter((r) => r.mode === 'inverse').length

  return (
    <div className="grid grid-cols-1 gap-5 xl:grid-cols-2">
      <Card title="Governing Equation, Domain & Pre-Flight Checks" icon={<Sigma size={17} className="text-royal" />} right={<Badge color="#10B981"><ShieldCheck size={12} /> SymPy parse OK</Badge>}>
        <div className="space-y-4">
          <div>
            <div className="label mb-1.5">LaTeX equation editor</div>
            <div className="rounded-2xl border border-violet-100 bg-gradient-to-br from-violet-50/70 to-white p-3">
              <Equation latex={latex} className="overflow-x-auto py-1 text-center text-[1.15rem]" />
            </div>
            <textarea
              value={latex}
              onChange={(e) => setLatex(e.target.value)}
              rows={2}
              spellCheck={false}
              className="input mt-2 resize-none text-xs"
            />
          </div>

          <div className="grid grid-cols-2 gap-3">
            {([
              [exp.isODE ? 'State bounds x' : 'Spatial domain x', 'xMin', 'xMax'],
              [exp.isODE ? 'Time domain t' : 'Temporal domain t', 'tMin', 'tMax'],
            ] as const).map(([lab, lo, hi]) => (
              <div key={lab}>
                <div className="label mb-1.5">{lab}</div>
                <div className="flex items-center gap-1.5">
                  <span className="mono text-sm text-violet-400">[</span>
                  <input className="input" type="number" value={dom[lo]} onChange={(e) => setDom({ ...dom, [lo]: num(e.target.value, dom[lo]) })} />
                  <span className="mono text-sm text-violet-400">,</span>
                  <input className="input" type="number" value={dom[hi]} onChange={(e) => setDom({ ...dom, [hi]: num(e.target.value, dom[hi]) })} />
                  <span className="mono text-sm text-violet-400">]</span>
                </div>
              </div>
            ))}
          </div>

          <div>
            <div className="mb-1.5 flex items-center justify-between">
              <div className="label">Initial & boundary conditions</div>
              <button
                className="flex items-center gap-1 text-xs font-semibold text-royal hover:underline"
                onClick={() => setConds((c) => [...c, { id: `c${Date.now()}`, role: 'Boundary', kind: 'Neumann', where: 'x = 0', expr: 'uₓ(0, t) = 0' }])}
              >
                <Plus size={13} /> Add
              </button>
            </div>
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
              {conds.map((c) => (
                <div key={c.id} className="rounded-2xl border bg-white/80 p-2.5" style={{ borderColor: `${BC_COLORS[c.kind]}40` }}>
                  <div className="mb-1.5 flex items-center justify-between gap-2">
                    <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">{c.role} · <span className="mono normal-case">{c.where}</span></span>
                    <div className="flex items-center gap-1">
                      <select
                        value={c.kind}
                        onChange={(e) => setConds((cs) => cs.map((x) => (x.id === c.id ? { ...x, kind: e.target.value as BCKind } : x)))}
                        className="rounded-full px-2 py-0.5 text-[11px] font-bold outline-none"
                        style={{ color: BC_COLORS[c.kind], background: `${BC_COLORS[c.kind]}14`, border: `1px solid ${BC_COLORS[c.kind]}33` }}
                      >
                        {(['Dirichlet', 'Neumann', 'Periodic'] as BCKind[]).map((k) => <option key={k}>{k}</option>)}
                      </select>
                      <button aria-label="Remove condition" onClick={() => setConds((cs) => cs.filter((x) => x.id !== c.id))} className="text-slate-300 hover:text-[#EC4899]"><X size={14} /></button>
                    </div>
                  </div>
                  <input
                    value={c.expr}
                    onChange={(e) => setConds((cs) => cs.map((x) => (x.id === c.id ? { ...x, expr: e.target.value } : x)))}
                    className="w-full bg-transparent font-mono text-[13px] text-indigo-950 outline-none"
                  />
                </div>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <div>
              <div className="label mb-1.5">Physical invariant</div>
              <div className="flex flex-wrap gap-1.5">
                {INVARIANTS.map((i) => (
                  <button key={i} onClick={() => setInv(i)} className={cx('rounded-full border px-3 py-1 text-xs font-semibold transition', inv === i ? 'border-royal bg-royal text-white' : 'border-violet-100 bg-white/80 text-violet-700 hover:bg-violet-50')}>
                    {i}
                  </button>
                ))}
              </div>
            </div>
            <div>
              <div className="mb-1.5 flex items-center justify-between">
                <div className="label">Auto-normalize to [−1, 1]</div>
                <Switch checked={norm} onChange={setNorm} />
              </div>
              <div className={cx('mono space-y-0.5 rounded-xl border border-dashed p-2 text-[11px] transition', norm ? 'border-turq/50 bg-cyan-50/60 text-cyan-900' : 'border-slate-200 bg-slate-50 text-slate-400')}>
                <div><Ruler size={11} className="mr-1 inline" />x′ = {(2 / xr).toPrecision(3)}·x {(-(dom.xMax + dom.xMin) / xr) >= 0 ? '+' : '−'} {Math.abs((dom.xMax + dom.xMin) / xr).toPrecision(3)}</div>
                <div><Ruler size={11} className="mr-1 inline" />t′ = {(2 / tr).toPrecision(3)}·t {(-(dom.tMax + dom.tMin) / tr) >= 0 ? '+' : '−'} {Math.abs((dom.tMax + dom.tMin) / tr).toPrecision(3)}</div>
                <div><Ruler size={11} className="mr-1 inline" />u′ = u / {exp.uScale.toFixed(2)}</div>
              </div>
            </div>
          </div>
        </div>
      </Card>

      <div className="flex flex-col gap-5">
        <Card
          title="Dynamic Equation Parameters & Coefficients"
          icon={<Layers size={17} className="text-turq" />}
          right={<Badge color="#06B6D4">{rows.length} detected · {inverseCount} inverse</Badge>}
        >
          <div className="space-y-3">
            {rows.length === 0 && <p className="text-sm text-slate-500">No known coefficient symbols detected in the equation.</p>}
            {rows.map((p) => (
              <ParamRow key={p.symbol} p={p} onChange={(patch) => setParam(p, patch)} />
            ))}
          </div>
        </Card>
        <div className="flex justify-end">
          <button className="btn-primary px-7 py-3 text-[15px]" onClick={onRun}>
            <Play size={17} fill="currentColor" /> Run Autonomous Agent Workflow
          </button>
        </div>
      </div>
    </div>
  )
}

function ParamRow({ p, onChange }: { p: ParamDef; onChange: (patch: Partial<ParamDef>) => void }) {
  const inverse = p.mode === 'inverse'
  return (
    <div className="rounded-2xl border border-violet-100/80 bg-white/70 p-3">
      <div className="mb-2 flex items-center justify-between gap-2">
        <div className="flex items-center gap-2.5">
          <span className="mono flex h-8 w-8 items-center justify-center rounded-xl bg-violet-100 text-base font-semibold text-royal">{p.label}</span>
          <div>
            <div className="mono text-[13px] font-semibold text-indigo-950">{p.symbol}</div>
            <div className="text-[11px] text-slate-500">{p.description}</div>
          </div>
        </div>
        <Segmented
          small
          value={p.mode}
          onChange={(m) => onChange({ mode: m })}
          options={[{ value: 'fixed', label: 'Fixed Value' }, { value: 'inverse', label: 'Inverse / Unknown' }]}
        />
      </div>
      {!inverse ? (
        <div className="flex items-center gap-3">
          <input type="range" min={p.min} max={p.max} step={p.step} value={p.value} onChange={(e) => onChange({ value: parseFloat(e.target.value) })} className="h-1.5 flex-1 cursor-pointer accent-[#7C3AED]" />
          <input type="number" step={p.step} value={p.value} onChange={(e) => onChange({ value: parseFloat(e.target.value) || 0 })} className="input !w-24" />
        </div>
      ) : (
        <div className="space-y-2.5">
          <div className="grid grid-cols-3 gap-2">
            {([['Initial guess', 'guess'], ['Search min', 'lo'], ['Search max', 'hi']] as const).map(([lab, key]) => (
              <label key={key} className="block">
                <span className="text-[10px] font-semibold uppercase tracking-wide text-slate-400">{lab}</span>
                <input type="number" step={p.step} value={p[key]} onChange={(e) => onChange({ [key]: parseFloat(e.target.value) || 0 } as Partial<ParamDef>)} className="input" />
              </label>
            ))}
          </div>
          <div className="flex items-center justify-between">
            <Switch checked={p.ensemble} onChange={(v) => onChange({ ensemble: v })} label="Ensemble UQ (n=8)" />
            {p.truth !== undefined && <span className="mono text-[10px] text-slate-400">hidden truth withheld from solver</span>}
          </div>
          <Dropzone file={p.sensorFile} onFile={(f) => onChange({ sensorFile: f })} />
        </div>
      )}
    </div>
  )
}

function Dropzone({ file, onFile }: { file: string | null; onFile: (name: string | null) => void }) {
  const input = useRef<HTMLInputElement>(null)
  const [over, setOver] = useState(false)
  const [rows, setRows] = useState<number | null>(null)
  const read = (f: File) => {
    onFile(f.name)
    f.text().then((t) => setRows(Math.max(0, t.split(/\r?\n/).filter(Boolean).length - 1))).catch(() => setRows(null))
  }
  return (
    <div
      onClick={() => input.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setOver(true) }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); const f = e.dataTransfer.files?.[0]; if (f) read(f) }}
      className={cx('flex cursor-pointer items-center gap-3 rounded-xl border-2 border-dashed px-3 py-2 text-xs transition', over ? 'border-royal bg-violet-50' : 'border-violet-200 bg-violet-50/40 hover:bg-violet-50')}
    >
      <input ref={input} type="file" accept=".csv,text/csv" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) read(f) }} />
      {file ? (
        <>
          <FileSpreadsheet size={18} className="text-emer" />
          <div className="min-w-0 flex-1"><div className="mono truncate font-semibold text-indigo-950">{file}</div><div className="text-slate-500">{rows !== null ? `${rows} sensor rows parsed` : 'sensor data attached'}</div></div>
          <button onClick={(e) => { e.stopPropagation(); onFile(null); setRows(null) }} className="text-slate-400 hover:text-[#EC4899]"><X size={14} /></button>
        </>
      ) : (
        <>
          <Upload size={18} className="text-royal" />
          <span className="text-violet-700"><b>Drop CSV sensor data</b> or click to browse <span className="text-slate-400">(x, t, u)</span></span>
        </>
      )}
    </div>
  )
}

