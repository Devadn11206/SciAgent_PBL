import { useEffect, useMemo, useRef, useState } from 'react'
import { AlertCircle, Battery, CheckCircle, Download, FileSpreadsheet, FlaskConical, Layers, Loader2, Play, Plus, Radio, Ruler, ShieldCheck, Sigma, Sliders, Sparkles, Upload, Wand2, Waves, Wind, X } from 'lucide-react'
import type { BCKind, ConditionDef, ConditionIR, Invariant, ParamDef, ProblemIR } from '@/data/types'
import { PARAM_CATALOG, PRESET_METAS } from '@/data/presetsCatalog'
import { Equation } from '@/components/Equation'
import { Badge, Card, cx, Segmented, Switch } from '@/components/ui'
import { fetchPresets, fetchSampleSensorCsv, generateProblemFromNL, parseEquation, uploadSensorCsv, type ParseEquationResponse } from '@/services/api'

const ICONS = { waves: Waves, battery: Battery, fiber: Radio, wing: Wind }

const detect = (latex: string) =>
  Object.values(PARAM_CATALOG).filter((c) => (c.symbol === 'eps' ? /\\varepsilon|\\epsilon/.test(latex) : latex.includes(c.token)))

const BC_COLORS: Record<BCKind, string> = { Dirichlet: '#7C3AED', Neumann: '#06B6D4', Periodic: '#EC4899' }
const INVARIANTS: Invariant[] = ['None', 'L2 Energy Conservation', 'Mass Conservation']

interface Props {
  activeProblem: ProblemIR
  presets: ProblemIR[]
  onSelectPreset: (problem: ProblemIR) => void
  onRunWorkflow: (problem: ProblemIR) => void
}

export function StudioView({ activeProblem, presets, onSelectPreset, onRunWorkflow }: Props) {
  const [nlPrompt, setNlPrompt] = useState('')
  const [nlLoading, setNlLoading] = useState(false)
  const [nlNotice, setNlNotice] = useState<string | null>(null)

  const handleGenerateFromNL = async () => {
    if (!nlPrompt.trim()) return
    setNlLoading(true)
    setNlNotice(null)
    try {
      const generated = await generateProblemFromNL(nlPrompt)
      onSelectPreset(generated)
      setNlNotice(`Generated "${generated.name}" with canonical domain & parameters!`)
    } catch (e: any) {
      setNlNotice(`Copilot notice: ${e.message}`)
    } finally {
      setNlLoading(false)
    }
  }

  return (
    <div className="space-y-5">
      {/* Natural-Language "Describe Your System" Copilot Bar */}
      <div className="relative overflow-hidden rounded-3xl border border-violet-200/80 bg-gradient-to-r from-violet-100/80 via-white/90 to-pink-100/80 p-4 shadow-glass backdrop-blur-xl">
        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div className="flex items-center gap-2.5">
            <div className="flex h-9 w-9 items-center justify-center rounded-2xl bg-royal text-white shadow-md">
              <Wand2 size={18} />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-sm font-bold text-indigo-950">Natural-Language Equation Copilot</span>
                <Badge color="#7C3AED" className="text-[10px]">Local Ollama / SymPy</Badge>
              </div>
              <p className="text-[11px] text-slate-500">Describe any dynamical system in plain English to auto-populate equation, domain & boundaries</p>
            </div>
          </div>
          <div className="flex flex-1 max-w-xl items-center gap-2">
            <input
              type="text"
              value={nlPrompt}
              onChange={(e) => setNlPrompt(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleGenerateFromNL()}
              placeholder="e.g. 1D damped nonlinear string with weak viscosity or Duffing chaotic oscillator..."
              className="flex-1 rounded-2xl border border-violet-200 bg-white/90 px-4 py-2 text-xs text-indigo-950 placeholder-slate-400 shadow-inner focus:border-royal focus:outline-none"
            />
            <button
              disabled={!nlPrompt.trim() || nlLoading}
              onClick={handleGenerateFromNL}
              className="flex items-center gap-1.5 shrink-0 rounded-2xl bg-royal px-4 py-2 text-xs font-bold text-white shadow-md transition hover:bg-violet-700 disabled:opacity-50"
            >
              {nlLoading ? <Loader2 size={13} className="animate-spin" /> : <Sparkles size={13} />}
              Generate Equation & Domain
            </button>
          </div>
        </div>
        {nlNotice && (
          <div className="mt-2 text-center text-xs font-semibold text-royal animate-fadeIn">
            ✨ {nlNotice}
          </div>
        )}
      </div>

      <div>
        <div className="mb-2 flex items-center justify-between">
          <p className="label flex items-center gap-2"><FlaskConical size={13} /> Real-World Nonlinear Case Study — one click</p>
          <Badge color="#10B981">
            ● Backend Live (Port 8080)
          </Badge>
        </div>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
          {presets.map((p) => {
            const meta = PRESET_METAS[p.id] || {
              id: p.id,
              name: p.name,
              title: p.name,
              subtitle: 'Canonical Nonlinear Problem IR',
              tint: 'from-violet-50/70 to-purple-50/40',
              accent: '#7C3AED',
              icon: 'waves' as const,
              mode: 'Forward' as const,
              latex: p.equation_latex,
              referenceSolver: 'High-Order Solver',
            }
            const Icon = ICONS[meta.icon] || Waves
            const active = p.id === activeProblem.id
            return (
              <button
                key={p.id}
                onClick={() => onSelectPreset(p)}
                className={cx(
                  'group relative overflow-hidden rounded-3xl border bg-gradient-to-br p-4 text-left shadow-glass backdrop-blur-md transition hover:-translate-y-0.5',
                  meta.tint,
                  active ? 'border-transparent ring-2' : 'border-white',
                )}
                style={active ? ({ ['--tw-ring-color' as string]: meta.accent } as React.CSSProperties) : undefined}
              >
                <div className="mb-2 flex items-center justify-between">
                  <span className="flex h-9 w-9 items-center justify-center rounded-2xl" style={{ background: `${meta.accent}1f`, color: meta.accent }}><Icon size={18} /></span>
                  <Badge color={meta.accent}>{meta.mode}</Badge>
                </div>
                <div className="text-[15px] font-bold leading-tight text-indigo-950">{meta.title}</div>
                <div className="mb-2 text-xs font-medium text-violet-600">{meta.subtitle}</div>
                <div className="overflow-hidden rounded-xl bg-white/70 px-2 py-1 text-[11px]"><Equation latex={p.equation_latex} display={false} className="whitespace-nowrap [&_.katex]:text-[0.95em]" /></div>
              </button>
            )
          })}
        </div>
      </div>
      {/* keyed remount resets editable studio state when a preset changes */}
      <StudioBody key={activeProblem.id} problem={activeProblem} onRun={onRunWorkflow} />
    </div>
  )
}

function StudioBody({ problem, onRun }: { problem: ProblemIR; onRun: (prob: ProblemIR) => void }) {
  const [latex, setLatex] = useState(problem.equation_latex)
  const [dom, setDom] = useState({
    xMin: problem.domain.x?.min_val ?? -1.0,
    xMax: problem.domain.x?.max_val ?? 1.0,
    tMin: problem.domain.t.min_val ?? 0.0,
    tMax: problem.domain.t.max_val ?? 1.0,
  })

  const [conds, setConds] = useState<ConditionDef[]>(() => {
    const list: ConditionDef[] = []
    if (problem.initial_conditions) {
      problem.initial_conditions.forEach((ic: any, i: number) => {
        list.push({
          id: ic.id || `ic_${i}`,
          role: 'Initial',
          kind: 'Dirichlet',
          where: ic.location || 't = 0',
          expr: ic.expression || 'u(x, 0) = 0',
        })
      })
    }
    if (problem.boundary_conditions) {
      problem.boundary_conditions.forEach((bc: any, i: number) => {
        list.push({
          id: bc.id || `bc_${i}`,
          role: 'Boundary',
          kind: (bc.kind as BCKind) || 'Dirichlet',
          where: bc.location || 'x = boundary',
          expr: bc.expression || 'u(boundary, t) = 0',
        })
      })
    }
    if (list.length === 0) {
      list.push({ id: 'c1', role: 'Initial', kind: 'Dirichlet', where: 't = 0', expr: 'u(x,0) = -sin(pi*x)' })
      list.push({ id: 'c2', role: 'Boundary', kind: 'Dirichlet', where: 'x = ±1', expr: 'u(±1,t) = 0' })
    }
    return list
  })

  const [inv, setInv] = useState<Invariant>(
    problem.conservation_invariant === 'l2_energy' ? 'L2 Energy Conservation' : problem.conservation_invariant === 'mass' ? 'Mass Conservation' : 'None'
  )
  const [norm, setNorm] = useState(true)

  const [params, setParams] = useState<Record<string, ParamDef>>(() => {
    const map: Record<string, ParamDef> = {}
    problem.parameters.forEach((p: any) => {
      const cat = PARAM_CATALOG[p.name]
      map[p.name] = {
        symbol: p.name,
        label: cat?.label || p.name,
        token: cat?.token || `\\${p.name}`,
        description: cat?.description || `Parameter ${p.name}`,
        mode: p.mode,
        value: p.value ?? cat?.guess ?? 0.01,
        min: p.bounds ? p.bounds[0] : cat?.min ?? 0.001,
        max: p.bounds ? p.bounds[1] : cat?.max ?? 1.0,
        step: cat?.step ?? 0.01,
        guess: p.initial_guess ?? cat?.guess ?? 0.01,
        lo: p.bounds ? p.bounds[0] : cat?.lo ?? 0.001,
        hi: p.bounds ? p.bounds[1] : cat?.hi ?? 1.0,
        ensemble: true,
        sensorFile: null,
      }
    })
    return map
  })

  const [parseResult, setParseResult] = useState<ParseEquationResponse | null>(null)
  const [parsing, setParsing] = useState(false)
  const [parseError, setParseError] = useState<string | null>(null)

  // Trigger real backend symbolic parsing when equation changes
  useEffect(() => {
    let active = true
    const timer = setTimeout(() => {
      setParsing(true)
      parseEquation(latex, problem.id)
        .then((res) => {
          if (!active) return
          setParseResult(res)
          setParseError(null)

          // Auto-adjust sensible domain bounds when problem type changes unless locked
          if (res.problem_type === 'ODE') {
            setDom((prev) => ({
              ...prev,
              tMin: prev.tMin ?? 0.0,
              tMax: prev.tMax > 5.0 ? prev.tMax : 20.0,
            }))
          } else if (res.problem_type === 'PDE') {
            setDom((prev) => ({
              ...prev,
              xMin: prev.xMin !== undefined && prev.xMin < 0 ? prev.xMin : -1.0,
              xMax: prev.xMax !== undefined && prev.xMax > 0 ? prev.xMax : 1.0,
              tMin: 0.0,
              tMax: prev.tMax <= 2.0 ? prev.tMax : 1.0,
            }))
          }
        })
        .catch((err) => {
          if (!active) return
          setParseError(err.message)
        })
        .finally(() => {
          if (active) setParsing(false)
        })
    }, 450)
    return () => {
      active = false
      clearTimeout(timer)
    }
  }, [latex, problem.id])

  const rows = useMemo(() => {
    const list = detect(latex)
    return list.map((c) => params[c.symbol] ?? { ...c, value: c.min + (c.max - c.min) * 0.5, mode: 'fixed' as const, ensemble: true, sensorFile: null })
  }, [latex, params])

  const setParam = (p: ParamDef, patch: Partial<ParamDef>) =>
    setParams((prev) => ({ ...prev, [p.symbol]: { ...(prev[p.symbol] ?? p), ...patch } }))

  const xr = Math.max(1e-6, dom.xMax - dom.xMin)
  const tr = Math.max(1e-6, dom.tMax - dom.tMin)
  const inverseCount = rows.filter((r) => r.mode === 'inverse').length

  const handleStartWorkflow = async () => {
    let currentParse = parseResult
    try {
      currentParse = await parseEquation(latex, problem.id)
      setParseResult(currentParse)
    } catch (e) {
      console.warn('Pre-run parse notice:', e)
    }

    const isPde = currentParse?.problem_type === 'PDE' || (!currentParse && problem.problem_type === 'PDE')

    // Determine canonical problem id matching equation classification
    let computedId = problem.id
    const lowLatex = latex.toLowerCase()
    if (lowLatex.includes('\\nu') || lowLatex.includes('nu') || lowLatex.includes('burgers')) {
      computedId = 'burgers'
    } else if (lowLatex.includes('\\epsilon') || lowLatex.includes('eps') || lowLatex.includes('allen') || lowLatex.includes('cahn') || lowLatex.includes('1 - u^2')) {
      computedId = 'allen-cahn'
    } else if (lowLatex.includes('\\psi') || lowLatex.includes('nlse') || lowLatex.includes('schrodinger')) {
      computedId = 'nlse'
    } else if (lowLatex.includes('\\ddot') || lowLatex.includes('d^2 x') || !isPde) {
      computedId = 'duffing'
    }

    const updatedProblem: ProblemIR = {
      ...problem,
      id: computedId,
      name: `${computedId.toUpperCase()} Custom System`,
      problem_type: isPde ? 'PDE' : 'ODE',
      equation_latex: latex,
      parameters: rows.map((r) => ({
        name: r.symbol,
        mode: r.mode,
        value: r.mode === 'fixed' ? r.value : null,
        initial_guess: r.mode === 'inverse' ? r.guess : null,
        bounds: r.mode === 'inverse' ? [r.lo, r.hi] : null,
      })),
      domain: {
        t: { min_val: dom.tMin, max_val: dom.tMax },
        x: isPde ? { min_val: dom.xMin, max_val: dom.xMax } : null,
      },
      conservation_invariant: inv === 'L2 Energy Conservation' ? 'l2_energy' : inv === 'Mass Conservation' ? 'mass' : 'none',
    }
    onRun(updatedProblem)
  }

  return (
    <div className="grid grid-cols-1 gap-5 xl:grid-cols-3">
      <Card
        className="xl:col-span-2"
        title="Symbolic Equation & Domain Specification"
        icon={<Sigma size={17} className="text-royal" />}
        right={
          <div className="flex items-center gap-2">
            {parsing && <span className="mono text-[10px] text-royal animate-pulse">Parsing SymPy…</span>}
            {parseResult && (
              <Badge color="#7C3AED">
                {parseResult.primary_regime.toUpperCase()}
              </Badge>
            )}
            <Badge color="#EC4899">{parseResult?.problem_type ?? problem.problem_type}</Badge>
          </div>
        }
      >
        <div className="space-y-4">
          <div>
            <div className="mb-1 flex items-center justify-between">
              <span className="label">Canonical LaTeX equation</span>
              {parseResult && (
                <span className="mono text-[10px] text-slate-500">
                  order: {parseResult.spatial_order}x, {parseResult.time_order}t
                </span>
              )}
            </div>
            <textarea
              value={latex}
              onChange={(e) => setLatex(e.target.value)}
              rows={2}
              className="mono w-full rounded-2xl border border-violet-100 bg-white/90 p-3 text-sm text-indigo-950 shadow-inner focus:border-royal focus:outline-none"
            />
          </div>

          {parseError && (
            <div className="flex items-center gap-2 rounded-xl bg-amber-50 p-2.5 text-xs text-amber-800">
              <AlertCircle size={14} /> <span>SymPy notice: {parseError}</span>
            </div>
          )}

          <div className="overflow-hidden rounded-2xl border border-violet-100 bg-gradient-to-r from-violet-50/70 via-white to-pink-50/70 p-4">
            <div className="label mb-2 flex items-center justify-between">
              <span>Interactive KaTeX render</span>
              {parseResult?.well_posedness.is_well_posed && (
                <span className="flex items-center gap-1 text-[11px] font-semibold text-emerald-600">
                  <CheckCircle size={12} /> Well-Posed Equation
                </span>
              )}
            </div>
            <div className="overflow-x-auto py-2"><Equation latex={latex} className="text-xl" /></div>
          </div>

          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            {([
              ['xMin', 'x min', dom.xMin],
              ['xMax', 'x max', dom.xMax],
              ['tMin', 't min', dom.tMin],
              ['tMax', 't max', dom.tMax],
            ] as const).map(([k, lab, val]) => (
              <label key={k} className="block">
                <span className="label">{lab}</span>
                <input
                  type="number"
                  step={0.1}
                  value={val}
                  onChange={(e) => setDom({ ...dom, [k]: parseFloat(e.target.value) || 0 })}
                  className="input mono"
                />
              </label>
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
                {parseResult?.scaling_factors && (
                  <div className="pt-1 text-[10px] text-cyan-700">
                    chain-rule derivative scaling: dx={parseResult.scaling_factors.dx?.toPrecision(3) ?? '1.0'}, dt={parseResult.scaling_factors.dt?.toPrecision(3) ?? '2.0'}
                  </div>
                )}
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
              <ParamRow key={p.symbol} p={p} expId={problem.id} onChange={(patch) => setParam(p, patch)} />
            ))}
          </div>
        </Card>
        <div className="flex justify-end">
          <button className="btn-primary px-7 py-3 text-[15px]" onClick={handleStartWorkflow}>
            <Play size={17} fill="currentColor" /> Run Autonomous Agent Workflow
          </button>
        </div>
      </div>
    </div>
  )
}

function ParamRow({ p, expId, onChange }: { p: ParamDef; expId: string; onChange: (patch: Partial<ParamDef>) => void }) {
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
          <Dropzone
            presetId={expId}
            file={p.sensorFile}
            onFile={(f) => onChange({ sensorFile: f })}
          />
        </div>
      )}
    </div>
  )
}

function Dropzone({
  presetId,
  file,
  onFile,
}: {
  presetId: string
  file: string | null
  onFile: (name: string | null) => void
}) {
  const input = useRef<HTMLInputElement>(null)
  const [over, setOver] = useState(false)
  const [rows, setRows] = useState<number | null>(null)
  const [uploading, setUploading] = useState(false)
  const [uploadStatus, setUploadStatus] = useState<string | null>(null)

  // Interactive Sensor Noise & Sparsity Stress-Tester Sliders
  const [probeCount, setProbeCount] = useState<number>(80)
  const [noisePct, setNoisePct] = useState<number>(1.5)

  const handleUpload = async (f: File) => {
    setUploading(true)
    setUploadStatus(null)
    try {
      const res = await uploadSensorCsv(f)
      onFile(f.name)
      setRows(res.row_count)
      setUploadStatus(`${res.row_count} rows (${res.kind}) verified on backend`)
    } catch (err: any) {
      onFile(f.name)
      f.text().then((t) => setRows(Math.max(0, t.split(/\r?\n/).filter(Boolean).length - 1))).catch(() => setRows(null))
      setUploadStatus(err.message || 'Attached locally')
    } finally {
      setUploading(false)
    }
  }

  const loadSample = async (e: React.MouseEvent) => {
    e.stopPropagation()
    setUploading(true)
    try {
      // Use configured probeCount and noisePct for stress testing
      const resp = await fetch(
        `http://localhost:8080/api/sensors/sample/${presetId}?n_probes=${probeCount}&noise_pct=${noisePct}`
      )
      const csvText = await resp.text()
      const blob = new Blob([csvText], { type: 'text/csv' })
      const sampleFile = new File([blob], `synthetic_sensors_${presetId}_N${probeCount}_noise${noisePct}pct.csv`, { type: 'text/csv' })
      await handleUpload(sampleFile)
    } catch {
      onFile(`synthetic_sensors_${presetId}.csv`)
      setRows(probeCount)
      setUploadStatus(`Sample loaded (N=${probeCount}, noise=${noisePct}%)`)
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="space-y-3">
      {/* Sensor Noise & Sparsity Stress-Tester Controls */}
      <div className="rounded-2xl border border-violet-100 bg-violet-50/50 p-3 text-xs">
        <div className="mb-2 flex items-center justify-between">
          <span className="flex items-center gap-1.5 font-bold text-indigo-950">
            <Sliders size={13} className="text-royal" /> Sensor Noise & Sparsity Stress-Tester
          </span>
          <span className="mono text-[10px] text-violet-600 font-semibold">
            N = {probeCount} probes · ±{noisePct.toFixed(1)}% Noise
          </span>
        </div>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <div>
            <div className="flex justify-between text-[10px] font-semibold text-slate-500 mb-1">
              <span>Probe Sparsity</span>
              <span className="mono font-bold text-royal">{probeCount}</span>
            </div>
            <input
              type="range"
              min={10}
              max={500}
              step={10}
              value={probeCount}
              onChange={(e) => setProbeCount(parseInt(e.target.value))}
              className="h-1.5 w-full cursor-pointer accent-[#7C3AED]"
            />
          </div>
          <div>
            <div className="flex justify-between text-[10px] font-semibold text-slate-500 mb-1">
              <span>Gaussian Noise</span>
              <span className="mono font-bold text-rose-600">{noisePct.toFixed(1)}%</span>
            </div>
            <input
              type="range"
              min={0.0}
              max={15.0}
              step={0.5}
              value={noisePct}
              onChange={(e) => setNoisePct(parseFloat(e.target.value))}
              className="h-1.5 w-full cursor-pointer accent-[#EC4899]"
            />
          </div>
        </div>
      </div>

      <div
        onClick={() => input.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setOver(true) }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => { e.preventDefault(); setOver(false); const f = e.dataTransfer.files?.[0]; if (f) handleUpload(f) }}
        className={cx('flex cursor-pointer items-center gap-3 rounded-xl border-2 border-dashed px-3 py-2 text-xs transition', over ? 'border-royal bg-violet-50' : 'border-violet-200 bg-violet-50/40 hover:bg-violet-50')}
      >
        <input ref={input} type="file" accept=".csv,text/csv" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) handleUpload(f) }} />
        {file ? (
          <>
            <FileSpreadsheet size={18} className="text-emer" />
            <div className="min-w-0 flex-1">
              <div className="mono truncate font-semibold text-indigo-950">{file}</div>
              <div className="text-slate-500">{uploadStatus || (rows !== null ? `${rows} sensor rows` : 'sensor data attached')}</div>
            </div>
            <button onClick={(e) => { e.stopPropagation(); onFile(null); setRows(null); setUploadStatus(null) }} className="text-slate-400 hover:text-[#EC4899]"><X size={14} /></button>
          </>
        ) : (
          <>
            <Upload size={18} className={uploading ? 'animate-bounce text-royal' : 'text-royal'} />
            <div className="min-w-0 flex-1">
              <span className="text-violet-700"><b>Drop CSV sensor data</b> or click to upload <span className="text-slate-400">(x, t, u)</span></span>
            </div>
            <button
              onClick={loadSample}
              disabled={uploading}
              className="mono flex items-center gap-1 rounded-lg border border-violet-200 bg-white/90 px-2.5 py-1 text-[11px] font-semibold text-royal shadow-sm hover:bg-violet-50"
            >
              <Download size={11} /> Load Synthetic CSV (N={probeCount}, {noisePct}%)
            </button>
          </>
        )}
      </div>
    </div>
  )
}
