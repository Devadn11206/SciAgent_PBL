import { useEffect, useMemo, useRef, useState } from 'react'
import {
  AlertCircle,
  Battery,
  CheckCircle,
  Download,
  FileSpreadsheet,
  FlaskConical,
  Layers,
  Loader2,
  Play,
  Plus,
  Radio,
  Ruler,
  ShieldCheck,
  Sigma,
  Sliders,
  Sparkles,
  Upload,
  Wand2,
  Waves,
  Wind,
  X,
} from 'lucide-react'
import type { BCKind, ConditionDef, Invariant, ParamDef, ProblemIR } from '@/data/types'
import { PARAM_CATALOG, PRESET_METAS } from '@/data/presetsCatalog'
import { Equation } from '@/components/Equation'
import { Badge, Card, cx, Segmented, Switch } from '@/components/ui'
import {
  generateProblemFromNL,
  parseEquation,
  uploadSensorCsv,
  type ParameterDetected,
  type ParseEquationResponse,
} from '@/services/api'

const ICONS = { waves: Waves, battery: Battery, fiber: Radio, wing: Wind }

const BC_COLORS: Record<BCKind, string> = {
  Dirichlet: '#7C3AED',
  Neumann: '#06B6D4',
  Periodic: '#EC4899',
}
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

  // Quick preset chips definition
  const quickChips = useMemo(() => {
    const list = [
      { id: 'burgers', label: "Viscous Burgers'", color: '#DB2777' },
      { id: 'allen-cahn', label: 'Allen-Cahn', color: '#D97706' },
      { id: 'nlse', label: 'Cubic-Quintic NLSE', color: '#0891B2' },
      { id: 'duffing', label: 'Forced Duffing', color: '#7C3AED' },
      { id: 'kdv', label: 'KdV Soliton', color: '#059669' },
      { id: 'gray-scott-2d', label: '2D Gray-Scott', color: '#4F46E5' },
      { id: 'heat-3d', label: '3D Heat Diffusion', color: '#EA580C' },
    ]
    return list
  }, [])

  return (
    <div className="space-y-4">
      {/* Natural-Language "Describe Your System" Copilot Bar */}
      <div className="relative overflow-hidden rounded-3xl border border-violet-200/80 bg-gradient-to-r from-violet-100/80 via-white/90 to-pink-100/80 p-3.5 shadow-glass backdrop-blur-xl">
        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-2xl bg-royal text-white shadow-md">
              <Wand2 size={16} />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xs font-bold text-indigo-950">Natural-Language Equation Copilot</span>
                <Badge color="#7C3AED" className="text-[9px]">Local Ollama / SymPy</Badge>
              </div>
              <p className="text-[10px] text-slate-500">
                Describe any dynamical system in plain English to auto-populate equation, domain & boundaries
              </p>
            </div>
          </div>
          <div className="flex flex-1 max-w-xl items-center gap-2">
            <input
              type="text"
              value={nlPrompt}
              onChange={(e) => setNlPrompt(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleGenerateFromNL()}
              placeholder="e.g. 1D damped nonlinear string with weak viscosity or Duffing chaotic oscillator..."
              className="flex-1 rounded-2xl border border-violet-200 bg-white/90 px-3.5 py-1.5 text-xs text-indigo-950 placeholder-slate-400 shadow-inner focus:border-royal focus:outline-none"
            />
            <button
              disabled={!nlPrompt.trim() || nlLoading}
              onClick={handleGenerateFromNL}
              className="flex items-center gap-1.5 shrink-0 rounded-2xl bg-royal px-3.5 py-1.5 text-xs font-bold text-white shadow-md transition hover:bg-violet-700 disabled:opacity-50"
            >
              {nlLoading ? <Loader2 size={13} className="animate-spin" /> : <Sparkles size={13} />}
              Generate Equation & Domain
            </button>
          </div>
        </div>
        {nlNotice && (
          <div className="mt-1.5 text-center text-xs font-semibold text-royal animate-fadeIn">
            ✨ {nlNotice}
          </div>
        )}
      </div>

      {/* Main Studio Workspace with Quick-Load Chips Header */}
      <StudioBody
        key={activeProblem.id}
        problem={activeProblem}
        presets={presets}
        quickChips={quickChips}
        onSelectPreset={onSelectPreset}
        onRun={onRunWorkflow}
      />
    </div>
  )
}

function StudioBody({
  problem,
  presets,
  quickChips,
  onSelectPreset,
  onRun,
}: {
  problem: ProblemIR
  presets: ProblemIR[]
  quickChips: { id: string; label: string; color: string }[]
  onSelectPreset: (p: ProblemIR) => void
  onRun: (prob: ProblemIR) => void
}) {
  const [latex, setLatex] = useState(problem.equation_latex)
  const [dom, setDom] = useState({
    xMin: problem.domain.x?.min_val ?? -1.0,
    xMax: problem.domain.x?.max_val ?? 1.0,
    yMin: problem.domain.y?.min_val ?? -1.0,
    yMax: problem.domain.y?.max_val ?? 1.0,
    zMin: problem.domain.z?.min_val ?? -1.0,
    zMax: problem.domain.z?.max_val ?? 1.0,
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
      if (problem.problem_type === 'ODE') {
        list.push({ id: 'ic1', role: 'Initial', kind: 'Dirichlet', where: 't = 0', expr: 'x(0) = 0.5' })
        list.push({ id: 'ic2', role: 'Initial', kind: 'Dirichlet', where: 't = 0', expr: 'x_dot(0) = 0.0' })
      } else {
        list.push({ id: 'ic1', role: 'Initial', kind: 'Dirichlet', where: 't = 0', expr: 'u(x, 0) = -sin(pi*x)' })
        list.push({ id: 'bc1', role: 'Boundary', kind: 'Dirichlet', where: 'x = ±1', expr: 'u(±1, t) = 0' })
      }
    }
    return list
  })

  const [inv, setInv] = useState<Invariant>(
    problem.conservation_invariant === 'l2_energy'
      ? 'L2 Energy Conservation'
      : problem.conservation_invariant === 'mass'
      ? 'Mass Conservation'
      : 'None'
  )
  const [norm, setNorm] = useState(true)

  // Track discovered free parameters strictly from SymPy AST
  const [params, setParams] = useState<Record<string, ParamDef>>(() => {
    const map: Record<string, ParamDef> = {}
    problem.parameters.forEach((p: any) => {
      const cat = PARAM_CATALOG[p.name]
      map[p.name] = {
        symbol: p.name,
        label: p.symbol || cat?.label || p.name,
        token: p.latex_token || cat?.token || `\\${p.name}`,
        description: p.description || cat?.description || `Parameter ${p.name}`,
        mode: p.mode,
        value: p.value ?? cat?.guess ?? 1.0,
        min: p.bounds ? p.bounds[0] : cat?.min ?? 0.001,
        max: p.bounds ? p.bounds[1] : cat?.max ?? 5.0,
        step: cat?.step ?? 0.01,
        guess: p.initial_guess ?? cat?.guess ?? 1.0,
        lo: p.bounds ? p.bounds[0] : cat?.lo ?? 0.001,
        hi: p.bounds ? p.bounds[1] : cat?.hi ?? 5.0,
        ensemble: true,
        sensorFile: null,
      }
    })
    return map
  })

  // Unified Inverse Sensor Data & UQ State
  const [unifiedProbeCount, setUnifiedProbeCount] = useState<number>(80)
  const [unifiedNoisePct, setUnifiedNoisePct] = useState<number>(1.5)
  const [unifiedEnsemble, setUnifiedEnsemble] = useState<boolean>(true)
  const [unifiedSensorFile, setUnifiedSensorFile] = useState<string | null>(null)
  const [unifiedSensorRows, setUnifiedSensorRows] = useState<number | null>(null)

  const [parseResult, setParseResult] = useState<ParseEquationResponse | null>(null)
  const [parsing, setParsing] = useState(false)
  const [parseError, setParseError] = useState<string | null>(null)

  // 400ms Debounced live equation auto-detection
  useEffect(() => {
    let active = true
    const timer = setTimeout(() => {
      setParsing(true)
      parseEquation(latex, problem.id)
        .then((res) => {
          if (!active) return
          setParseResult(res)
          setParseError(null)

          // 1. Strictly replace parameters with ONLY the exact free coefficients detected in AST
          const newParams: Record<string, ParamDef> = {}
          res.detected_parameters.forEach((dp: ParameterDetected) => {
            const cat = PARAM_CATALOG[dp.name]
            const existing = params[dp.name]
            newParams[dp.name] = {
              symbol: dp.name,
              label: dp.symbol || cat?.label || dp.name,
              token: cat?.token || `\\${dp.name}`,
              description: dp.description || cat?.description || `Parameter ${dp.name}`,
              mode: existing ? existing.mode : dp.suggested_mode,
              value: existing?.value ?? dp.value ?? cat?.guess ?? 1.0,
              min: dp.bounds ? dp.bounds[0] : cat?.min ?? 0.001,
              max: dp.bounds ? dp.bounds[1] : cat?.max ?? 5.0,
              step: cat?.step ?? 0.01,
              guess: existing?.guess ?? dp.initial_guess ?? cat?.guess ?? 1.0,
              lo: dp.bounds ? dp.bounds[0] : cat?.lo ?? 0.001,
              hi: dp.bounds ? dp.bounds[1] : cat?.hi ?? 5.0,
              ensemble: true,
              sensorFile: null,
            }
          })
          setParams(newParams)

          // 2. Smart Domain, IC/BC & Invariant Auto-Configuration
          if (res.problem_type === 'ODE') {
            setDom((prev) => ({
              ...prev,
              tMin: 0.0,
              tMax: prev.tMax > 5.0 ? prev.tMax : 20.0,
            }))
            // Auto-configure ODE initial conditions
            setConds([
              { id: 'ic1', role: 'Initial', kind: 'Dirichlet', where: 't = 0', expr: 'x(0) = 0.5' },
              { id: 'ic2', role: 'Initial', kind: 'Dirichlet', where: 't = 0', expr: 'x_dot(0) = 0.0' },
            ])
            setInv('None')
          } else {
            // PDE detected
            const low = latex.toLowerCase()
            const isNlse = low.includes('\\psi') || low.includes('nlse') || low.includes('schrodinger')
            const isKdv = low.includes('kdv') || (low.includes('6') && low.includes('partial') && res.spatial_order >= 3)
            const isAllenCahn = low.includes('allen') || low.includes('cahn') || low.includes('1 - u^2') || low.includes('u^3 - u')

            if (isNlse) {
              setDom((prev) => ({ ...prev, xMin: -6.0, xMax: 6.0, tMin: 0.0, tMax: 1.0 }))
              setConds([
                { id: 'ic1', role: 'Initial', kind: 'Dirichlet', where: 't = 0', expr: 'psi(x, 0) = 1.5 * sech(1.5 * x)' },
                { id: 'bc1', role: 'Boundary', kind: 'Periodic', where: 'x = ±6', expr: 'psi(-6, t) = psi(+6, t)' },
              ])
              setInv('L2 Energy Conservation')
            } else if (isKdv) {
              setDom((prev) => ({ ...prev, xMin: -10.0, xMax: 10.0, tMin: 0.0, tMax: 1.0 }))
              setConds([
                { id: 'ic1', role: 'Initial', kind: 'Dirichlet', where: 't = 0', expr: 'u(x, 0) = 0.5 * sech(0.5 * x)^2' },
                { id: 'bc1', role: 'Boundary', kind: 'Periodic', where: 'x = ±10', expr: 'u(-10, t) = u(+10, t)' },
              ])
              setInv('Mass Conservation')
            } else if (isAllenCahn) {
              setDom((prev) => ({ ...prev, xMin: -1.0, xMax: 1.0, tMin: 0.0, tMax: 1.0 }))
              setConds([
                { id: 'ic1', role: 'Initial', kind: 'Dirichlet', where: 't = 0', expr: 'u(x, 0) = x^2 * cos(pi * x)' },
                { id: 'bc1', role: 'Boundary', kind: 'Dirichlet', where: 'x = ±1', expr: 'u(±1, t) = 0' },
              ])
              setInv('None')
            } else {
              // Default conservative or general PDE (e.g. Burgers)
              setDom((prev) => ({
                ...prev,
                xMin: prev.xMin !== undefined && prev.xMin < 0 ? prev.xMin : -1.0,
                xMax: prev.xMax !== undefined && prev.xMax > 0 ? prev.xMax : 1.0,
                tMin: 0.0,
                tMax: prev.tMax <= 2.0 ? prev.tMax : 1.0,
              }))
              setConds([
                { id: 'ic1', role: 'Initial', kind: 'Dirichlet', where: 't = 0', expr: 'u(x, 0) = -sin(pi * x)' },
                { id: 'bc1', role: 'Boundary', kind: 'Dirichlet', where: 'x = ±1', expr: 'u(±1, t) = 0' },
              ])
              // Set invariant based on pre-flight symbolic check or regime hints
              if (res.suggested_invariant === 'mass') {
                setInv('Mass Conservation')
              } else if (res.suggested_invariant === 'l2_energy') {
                setInv('L2 Energy Conservation')
              } else if (res.suggested_invariant === 'none') {
                setInv('None')
              } else {
                setInv(low.includes('nu') || low.includes('burgers') ? 'None' : 'Mass Conservation')
              }
            }
          }
        })
        .catch((err) => {
          if (!active) return
          setParseError(err.message)
        })
        .finally(() => {
          if (active) setParsing(false)
        })
    }, 400)
    return () => {
      active = false
      clearTimeout(timer)
    }
  }, [latex, problem.id])

  // Parameter list strictly derived from AST detected parameters
  const rows = useMemo(() => Object.values(params), [params])
  const inverseCount = useMemo(() => rows.filter((r) => r.mode === 'inverse').length, [rows])
  const isPde = parseResult ? parseResult.problem_type === 'PDE' : problem.problem_type === 'PDE'

  const setParam = (p: ParamDef, patch: Partial<ParamDef>) =>
    setParams((prev) => ({ ...prev, [p.symbol]: { ...(prev[p.symbol] ?? p), ...patch } }))

  const xr = Math.max(1e-6, dom.xMax - dom.xMin)
  const tr = Math.max(1e-6, dom.tMax - dom.tMin)

  const handleStartWorkflow = async () => {
    let currentParse = parseResult
    try {
      currentParse = await parseEquation(latex, problem.id)
      setParseResult(currentParse)
    } catch (e) {
      console.warn('Pre-run parse notice:', e)
    }

    const currentPde = currentParse?.problem_type === 'PDE' || (!currentParse && problem.problem_type === 'PDE')

    // Determine canonical problem id matching equation classification
    let computedId = problem.id
    const lowLatex = latex.toLowerCase()
    if (lowLatex.includes('\\nu') || lowLatex.includes('nu') || lowLatex.includes('burgers')) {
      computedId = 'burgers'
    } else if (
      lowLatex.includes('\\epsilon') ||
      lowLatex.includes('eps') ||
      lowLatex.includes('allen') ||
      lowLatex.includes('cahn') ||
      lowLatex.includes('1 - u^2') ||
      lowLatex.includes('u^3 - u')
    ) {
      computedId = 'allen-cahn'
    } else if (lowLatex.includes('\\psi') || lowLatex.includes('nlse') || lowLatex.includes('schrodinger')) {
      computedId = 'nlse'
    } else if (lowLatex.includes('kdv') || (lowLatex.includes('6') && currentPde && (currentParse?.spatial_order ?? 0) >= 3)) {
      computedId = 'kdv'
    } else if (lowLatex.includes('\\ddot') || lowLatex.includes('d^2 x') || !currentPde) {
      computedId = 'duffing'
    }

    const computedDim: 0 | 1 | 2 | 3 = !currentPde ? 0 : (currentParse?.spatial_dim ?? (problem.spatial_dim ?? 1)) as 0 | 1 | 2 | 3

    const updatedProblem: ProblemIR = {
      ...problem,
      id: computedId,
      name: `${computedId.toUpperCase()} System`,
      problem_type: currentPde ? 'PDE' : 'ODE',
      spatial_dim: computedDim,
      equation_latex: latex,
      derivatives: {
        spatial_order: currentParse?.spatial_order ?? (currentPde ? 2 : 0),
        time_order: currentParse?.time_order ?? (currentPde ? 1 : 2),
      },
      parameters: rows.map((r) => ({
        name: r.symbol,
        symbol: r.label || r.symbol,
        latex_token: r.token || `\\${r.symbol}`,
        description: r.description || `Parameter ${r.symbol}`,
        mode: r.mode,
        value: r.mode === 'fixed' ? (typeof r.value === 'number' && !isNaN(r.value) ? r.value : 1.0) : null,
        initial_guess: r.mode === 'inverse' ? (typeof r.guess === 'number' && !isNaN(r.guess) ? r.guess : 1.0) : null,
        bounds: r.mode === 'inverse' ? [r.lo ?? 0.01, r.hi ?? 5.0] : null,
      })),
      domain: {
        t: { min_val: dom.tMin, max_val: dom.tMax },
        x: currentPde ? { min_val: dom.xMin, max_val: dom.xMax } : null,
        y: currentPde && computedDim >= 2 ? { min_val: dom.yMin, max_val: dom.yMax } : null,
        z: currentPde && computedDim === 3 ? { min_val: dom.zMin, max_val: dom.zMax } : null,
      },
      initial_conditions: conds
        .filter((c) => c.role === 'Initial')
        .map((c) => ({
          id: c.id,
          role: 'Initial',
          location: c.where,
          expression: c.expr,
          order: c.expr.includes('dot') ? 1 : 0,
        })),
      boundary_conditions: currentPde
        ? conds
            .filter((c) => c.role === 'Boundary')
            .map((c) => ({
              id: c.id,
              role: 'Boundary',
              kind: c.kind,
              location: c.where,
              expression: c.expr,
              coordinate: c.kind === 'Periodic' ? 'periodic' : 'both',
            }))
        : [],
      conservation_invariant:
        inv === 'L2 Energy Conservation' ? 'l2_energy' : inv === 'Mass Conservation' ? 'mass' : 'none',
      metadata: {
        ...(problem.metadata || {}),
        sensor_probes: unifiedProbeCount,
        sensor_noise_pct: unifiedNoisePct,
        sensor_file: unifiedSensorFile,
        ensemble_uq: unifiedEnsemble,
      },
    }
    onRun(updatedProblem)
  }

  // Load preset cleanly from quick chips
  const handleQuickLoad = (chipId: string) => {
    const found = presets.find((p) => p.id === chipId)
    if (found) {
      onSelectPreset(found)
      return
    }
    // Fallback if not loaded in presets array
    const meta = PRESET_METAS[chipId]
    if (meta) {
      setLatex(meta.latex)
    }
  }

  return (
    <div className="grid grid-cols-1 gap-5 xl:grid-cols-3">
      {/* Left Column (2 cols): Symbolic Equation, Quick Chips, Domain & Conditions */}
      <Card
        className="xl:col-span-2"
        title="Symbolic Equation & Natural-Language Studio"
        icon={<Sigma size={17} className="text-royal" />}
        right={
          <div className="flex items-center gap-2">
            {parsing && <span className="mono text-[10px] text-royal animate-pulse">Parsing SymPy…</span>}
            {parseResult && (
              <Badge color="#7C3AED">
                {parseResult.primary_regime.toUpperCase()}
              </Badge>
            )}
            <Badge color={isPde ? '#EC4899' : '#8B5CF6'}>{isPde ? 'PDE' : 'ODE'}</Badge>
          </div>
        }
      >
        <div className="space-y-4">
          {/* Quick-Load Preset Chips */}
          <div className="flex flex-wrap items-center gap-2 border-b border-violet-100/80 pb-3">
            <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wider flex items-center gap-1.5 mr-1">
              <FlaskConical size={12} className="text-royal" /> Quick Presets:
            </span>
            {quickChips.map((chip) => {
              const active = problem.id === chip.id
              return (
                <button
                  key={chip.id}
                  onClick={() => handleQuickLoad(chip.id)}
                  className={cx(
                    'rounded-xl px-3 py-1 text-xs font-semibold transition flex items-center gap-1.5 shadow-sm',
                    active
                      ? 'bg-royal text-white shadow-md shadow-violet-200'
                      : 'border border-violet-100 bg-white/90 text-indigo-900 hover:bg-violet-50 hover:border-violet-200'
                  )}
                >
                  <span
                    className="h-2 w-2 rounded-full"
                    style={{ backgroundColor: active ? '#ffffff' : chip.color }}
                  />
                  {chip.label}
                </button>
              )
            })}
          </div>

          {/* Equation Input */}
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

          {/* Live KaTeX Render */}
          <div className="overflow-hidden rounded-2xl border border-violet-100 bg-gradient-to-r from-violet-50/70 via-white to-pink-50/70 p-4">
            <div className="label mb-2 flex items-center justify-between">
              <span>Interactive KaTeX render</span>
              <div className="flex items-center gap-2">
                {parseResult?.stability_advisory && (
                  <span className="flex items-center gap-1 rounded-full border border-amber-200 bg-amber-50 px-2 py-0.5 text-[10px] font-semibold text-amber-800">
                    <Sparkles size={11} className="text-amber-600" /> {parseResult.stability_advisory}
                  </span>
                )}
                {parseResult?.well_posedness.is_well_posed && (
                  <span className="flex items-center gap-1 text-[11px] font-semibold text-emerald-600">
                    <CheckCircle size={12} /> Well-Posed Equation
                  </span>
                )}
              </div>
            </div>
            <div className="overflow-x-auto py-2">
              <Equation latex={latex} className="text-xl" />
            </div>
          </div>

          {/* Smart Domain Bounds: Reveal X, Y, Z based on spatial_dim */}
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            {isPde && (
              <>
                <label className="block">
                  <span className="label">x min</span>
                  <input
                    type="number"
                    step={0.1}
                    value={dom.xMin}
                    onChange={(e) => setDom({ ...dom, xMin: parseFloat(e.target.value) || 0 })}
                    className="input mono"
                  />
                </label>
                <label className="block">
                  <span className="label">x max</span>
                  <input
                    type="number"
                    step={0.1}
                    value={dom.xMax}
                    onChange={(e) => setDom({ ...dom, xMax: parseFloat(e.target.value) || 0 })}
                    className="input mono"
                  />
                </label>
              </>
            )}
            {isPde && (parseResult?.spatial_dim ?? (problem.spatial_dim ?? 1)) >= 2 && (
              <>
                <label className="block">
                  <span className="label text-indigo-600 font-bold">y min (2D/3D)</span>
                  <input
                    type="number"
                    step={0.1}
                    value={dom.yMin}
                    onChange={(e) => setDom({ ...dom, yMin: parseFloat(e.target.value) || 0 })}
                    className="input mono"
                  />
                </label>
                <label className="block">
                  <span className="label text-indigo-600 font-bold">y max (2D/3D)</span>
                  <input
                    type="number"
                    step={0.1}
                    value={dom.yMax}
                    onChange={(e) => setDom({ ...dom, yMax: parseFloat(e.target.value) || 0 })}
                    className="input mono"
                  />
                </label>
              </>
            )}
            {isPde && (parseResult?.spatial_dim ?? (problem.spatial_dim ?? 1)) === 3 && (
              <>
                <label className="block">
                  <span className="label text-orange-600 font-bold">z min (3D)</span>
                  <input
                    type="number"
                    step={0.1}
                    value={dom.zMin}
                    onChange={(e) => setDom({ ...dom, zMin: parseFloat(e.target.value) || 0 })}
                    className="input mono"
                  />
                </label>
                <label className="block">
                  <span className="label text-orange-600 font-bold">z max (3D)</span>
                  <input
                    type="number"
                    step={0.1}
                    value={dom.zMax}
                    onChange={(e) => setDom({ ...dom, zMax: parseFloat(e.target.value) || 0 })}
                    className="input mono"
                  />
                </label>
              </>
            )}
            <label className="block">
              <span className="label">t min</span>
              <input
                type="number"
                step={0.1}
                value={dom.tMin}
                onChange={(e) => setDom({ ...dom, tMin: parseFloat(e.target.value) || 0 })}
                className="input mono"
              />
            </label>
            <label className="block">
              <span className="label">t max</span>
              <input
                type="number"
                step={0.1}
                value={dom.tMax}
                onChange={(e) => setDom({ ...dom, tMax: parseFloat(e.target.value) || 0 })}
                className="input mono"
              />
            </label>
          </div>

          {/* Initial & Boundary Conditions (Spatial BCs hidden if ODE) */}
          <div>
            <div className="mb-1.5 flex items-center justify-between">
              <div className="label">
                {isPde ? 'Initial & boundary conditions' : 'Initial conditions (Displacement & Velocity)'}
              </div>
              <button
                className="flex items-center gap-1 text-xs font-semibold text-royal hover:underline"
                onClick={() =>
                  setConds((c) => [
                    ...c,
                    isPde
                      ? { id: `c${Date.now()}`, role: 'Boundary', kind: 'Dirichlet', where: 'x = 0', expr: 'u(0, t) = 0' }
                      : { id: `c${Date.now()}`, role: 'Initial', kind: 'Dirichlet', where: 't = 0', expr: 'x(0) = 0.0' },
                  ])
                }
              >
                <Plus size={13} /> Add
              </button>
            </div>
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
              {conds.map((c) => (
                <div
                  key={c.id}
                  className="rounded-2xl border bg-white/80 p-2.5"
                  style={{ borderColor: `${BC_COLORS[c.kind] || '#7C3AED'}40` }}
                >
                  <div className="mb-1.5 flex items-center justify-between gap-2">
                    <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                      {c.role} · <span className="mono normal-case">{c.where}</span>
                    </span>
                    <div className="flex items-center gap-1">
                      {isPde && c.role === 'Boundary' && (
                        <select
                          value={c.kind}
                          onChange={(e) =>
                            setConds((cs) => cs.map((x) => (x.id === c.id ? { ...x, kind: e.target.value as BCKind } : x)))
                          }
                          className="rounded-full px-2 py-0.5 text-[11px] font-bold outline-none"
                          style={{
                            color: BC_COLORS[c.kind],
                            background: `${BC_COLORS[c.kind]}14`,
                            border: `1px solid ${BC_COLORS[c.kind]}33`,
                          }}
                        >
                          {(['Dirichlet', 'Neumann', 'Periodic'] as BCKind[]).map((k) => (
                            <option key={k}>{k}</option>
                          ))}
                        </select>
                      )}
                      <button
                        aria-label="Remove condition"
                        onClick={() => setConds((cs) => cs.filter((x) => x.id !== c.id))}
                        className="text-slate-300 hover:text-[#EC4899]"
                      >
                        <X size={14} />
                      </button>
                    </div>
                  </div>
                  <input
                    value={c.expr}
                    onChange={(e) =>
                      setConds((cs) => cs.map((x) => (x.id === c.id ? { ...x, expr: e.target.value } : x)))
                    }
                    className="w-full bg-transparent font-mono text-[13px] text-indigo-950 outline-none"
                  />
                </div>
              ))}
            </div>
          </div>

          {/* Physical Invariant & Normalization */}
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <div>
              <div className="label mb-1.5">Physical invariant</div>
              <div className="flex flex-wrap gap-1.5">
                {INVARIANTS.map((i) => (
                  <button
                    key={i}
                    onClick={() => setInv(i)}
                    className={cx(
                      'rounded-full border px-3 py-1 text-xs font-semibold transition',
                      inv === i
                        ? 'border-royal bg-royal text-white'
                        : 'border-violet-100 bg-white/80 text-violet-700 hover:bg-violet-50'
                    )}
                  >
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
              <div
                className={cx(
                  'mono space-y-0.5 rounded-xl border border-dashed p-2 text-[11px] transition',
                  norm ? 'border-turq/50 bg-cyan-50/60 text-cyan-900' : 'border-slate-200 bg-slate-50 text-slate-400'
                )}
              >
                {isPde && (
                  <div>
                    <Ruler size={11} className="mr-1 inline" />
                    x′ = {(2 / xr).toPrecision(3)}·x {(-(dom.xMax + dom.xMin) / xr) >= 0 ? '+' : '−'}{' '}
                    {Math.abs((dom.xMax + dom.xMin) / xr).toPrecision(3)}
                  </div>
                )}
                <div>
                  <Ruler size={11} className="mr-1 inline" />
                  t′ = {(2 / tr).toPrecision(3)}·t {(-(dom.tMax + dom.tMin) / tr) >= 0 ? '+' : '−'}{' '}
                  {Math.abs((dom.tMax + dom.tMin) / tr).toPrecision(3)}
                </div>
                {parseResult?.scaling_factors && (
                  <div className="pt-1 text-[10px] text-cyan-700">
                    chain-rule derivative scaling:{' '}
                    {isPde && `dx=${parseResult.scaling_factors.dx?.toPrecision(3) ?? '1.0'}, `}
                    dt={parseResult.scaling_factors.dt?.toPrecision(3) ?? '2.0'}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </Card>

      {/* Right Column: Parameters List + Single Unified Inverse/UQ Card */}
      <div className="flex flex-col gap-4">
        {/* Dynamic Parameters Card */}
        <Card
          title="Dynamic Equation Parameters"
          icon={<Layers size={17} className="text-turq" />}
          right={<Badge color="#06B6D4">{rows.length} detected · {inverseCount} inverse</Badge>}
        >
          <div className="space-y-3">
            {rows.length === 0 && (
              <p className="text-xs text-slate-500 py-2">
                No free symbolic parameter coefficients detected in this equation AST (pure mathematical dynamics).
              </p>
            )}
            {rows.map((p) => (
              <ParamRow key={p.symbol} p={p} onChange={(patch) => setParam(p, patch)} />
            ))}
          </div>
        </Card>

        {/* Unified Inverse Sensor Data & Ensemble UQ Card (Appears whenever >=1 parameter is inverse) */}
        {inverseCount > 0 && (
          <Card
            title="Inverse Sensor Data & Ensemble UQ"
            icon={<Sliders size={17} className="text-royal" />}
            right={<Badge color="#EC4899">Active for {inverseCount} Inverse Param(s)</Badge>}
          >
            <div className="space-y-3">
              {/* Ensemble UQ Toggle */}
              <div className="flex items-center justify-between rounded-xl bg-violet-50/60 p-2.5">
                <div>
                  <div className="text-xs font-bold text-indigo-950">Deep Ensemble UQ (n = 8)</div>
                  <div className="text-[10px] text-slate-500">
                    Compute epistemic parameter confidence intervals & posterior variance
                  </div>
                </div>
                <Switch checked={unifiedEnsemble} onChange={setUnifiedEnsemble} />
              </div>

              {/* Sensor Noise & Sparsity Stress-Tester Controls */}
              <div className="rounded-xl border border-violet-100 bg-violet-50/30 p-2.5 text-xs space-y-2">
                <div className="flex items-center justify-between">
                  <span className="font-bold text-indigo-950 text-[11px]">Sensor Noise & Sparsity</span>
                  <span className="mono text-[10px] font-semibold text-royal">
                    N={unifiedProbeCount} probes · ±{unifiedNoisePct.toFixed(1)}% Noise
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <div className="flex justify-between text-[10px] font-semibold text-slate-500 mb-0.5">
                      <span>Probes (Sparsity)</span>
                      <span className="mono font-bold text-royal">{unifiedProbeCount}</span>
                    </div>
                    <input
                      type="range"
                      min={10}
                      max={500}
                      step={10}
                      value={unifiedProbeCount}
                      onChange={(e) => setUnifiedProbeCount(parseInt(e.target.value))}
                      className="h-1.5 w-full cursor-pointer accent-[#7C3AED]"
                    />
                  </div>
                  <div>
                    <div className="flex justify-between text-[10px] font-semibold text-slate-500 mb-0.5">
                      <span>Gaussian Noise</span>
                      <span className="mono font-bold text-rose-600">{unifiedNoisePct.toFixed(1)}%</span>
                    </div>
                    <input
                      type="range"
                      min={0.0}
                      max={15.0}
                      step={0.5}
                      value={unifiedNoisePct}
                      onChange={(e) => setUnifiedNoisePct(parseFloat(e.target.value))}
                      className="h-1.5 w-full cursor-pointer accent-[#EC4899]"
                    />
                  </div>
                </div>
              </div>

              {/* Single Unified CSV Sensor Dropzone */}
              <UnifiedDropzone
                presetId={problem.id}
                probeCount={unifiedProbeCount}
                noisePct={unifiedNoisePct}
                file={unifiedSensorFile}
                rows={unifiedSensorRows}
                onFile={(fileName, rowCount) => {
                  setUnifiedSensorFile(fileName)
                  setUnifiedSensorRows(rowCount)
                }}
              />
            </div>
          </Card>
        )}

        {/* Action Button */}
        <div className="flex justify-end pt-1">
          <button className="btn-primary w-full py-3 text-[14px]" onClick={handleStartWorkflow}>
            <Play size={16} fill="currentColor" /> Run Autonomous Agent Workflow
          </button>
        </div>
      </div>
    </div>
  )
}

function ParamRow({
  p,
  onChange,
}: {
  p: ParamDef
  onChange: (patch: Partial<ParamDef>) => void
}) {
  const inverse = p.mode === 'inverse'
  return (
    <div className="rounded-2xl border border-violet-100/80 bg-white/70 p-3 shadow-xs">
      <div className="mb-2 flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="mono flex h-7 w-7 items-center justify-center rounded-xl bg-violet-100 text-sm font-semibold text-royal">
            {p.label}
          </span>
          <div>
            <div className="mono text-xs font-semibold text-indigo-950">{p.symbol}</div>
            <div className="text-[10px] text-slate-500 leading-tight">{p.description}</div>
          </div>
        </div>
        <Segmented
          small
          value={p.mode}
          onChange={(m) => onChange({ mode: m })}
          options={[
            { value: 'fixed', label: 'Fixed' },
            { value: 'inverse', label: 'Inverse' },
          ]}
        />
      </div>

      {!inverse ? (
        <div className="flex items-center gap-3">
          <input
            type="range"
            min={p.min}
            max={p.max}
            step={p.step}
            value={p.value}
            onChange={(e) => onChange({ value: parseFloat(e.target.value) })}
            className="h-1.5 flex-1 cursor-pointer accent-[#7C3AED]"
          />
          <input
            type="number"
            step={p.step}
            value={p.value}
            onChange={(e) => onChange({ value: parseFloat(e.target.value) || 0 })}
            className="input !w-20 !py-1 text-xs"
          />
        </div>
      ) : (
        <div className="grid grid-cols-3 gap-1.5 pt-1">
          {([
            ['Initial guess', 'guess'],
            ['Search min', 'lo'],
            ['Search max', 'hi'],
          ] as const).map(([lab, key]) => (
            <label key={key} className="block">
              <span className="text-[9px] font-semibold uppercase tracking-wide text-slate-400">{lab}</span>
              <input
                type="number"
                step={p.step}
                value={p[key]}
                onChange={(e) => onChange({ [key]: parseFloat(e.target.value) || 0 } as Partial<ParamDef>)}
                className="input !py-1 text-xs"
              />
            </label>
          ))}
        </div>
      )}
    </div>
  )
}

function UnifiedDropzone({
  presetId,
  probeCount,
  noisePct,
  file,
  rows,
  onFile,
}: {
  presetId: string
  probeCount: number
  noisePct: number
  file: string | null
  rows: number | null
  onFile: (name: string | null, rows: number | null) => void
}) {
  const input = useRef<HTMLInputElement>(null)
  const [over, setOver] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadStatus, setUploadStatus] = useState<string | null>(null)

  const handleUpload = async (f: File) => {
    setUploading(true)
    setUploadStatus(null)
    try {
      const res = await uploadSensorCsv(f)
      onFile(f.name, res.row_count)
      setUploadStatus(`${res.row_count} rows (${res.kind}) verified on backend`)
    } catch (err: any) {
      onFile(f.name, null)
      f.text()
        .then((t) => {
          const r = Math.max(0, t.split(/\r?\n/).filter(Boolean).length - 1)
          onFile(f.name, r)
        })
        .catch(() => {})
      setUploadStatus(err.message || 'Attached locally')
    } finally {
      setUploading(false)
    }
  }

  const loadSample = async (e: React.MouseEvent) => {
    e.stopPropagation()
    setUploading(true)
    try {
      const resp = await fetch(
        `http://localhost:8080/api/sensors/sample/${presetId}?n_probes=${probeCount}&noise_pct=${noisePct}`
      )
      const csvText = await resp.text()
      const blob = new Blob([csvText], { type: 'text/csv' })
      const sampleFile = new File(
        [blob],
        `synthetic_sensors_${presetId}_N${probeCount}_noise${noisePct}pct.csv`,
        { type: 'text/csv' }
      )
      await handleUpload(sampleFile)
    } catch {
      onFile(`synthetic_sensors_${presetId}.csv`, probeCount)
      setUploadStatus(`Sample loaded (N=${probeCount}, noise=${noisePct}%)`)
    } finally {
      setUploading(false)
    }
  }

  return (
    <div
      onClick={() => input.current?.click()}
      onDragOver={(e) => {
        e.preventDefault()
        setOver(true)
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault()
        setOver(false)
        const f = e.dataTransfer.files?.[0]
        if (f) handleUpload(f)
      }}
      className={cx(
        'flex cursor-pointer items-center gap-2.5 rounded-xl border-2 border-dashed px-3 py-2 text-xs transition',
        over ? 'border-royal bg-violet-50' : 'border-violet-200 bg-violet-50/40 hover:bg-violet-50'
      )}
    >
      <input
        ref={input}
        type="file"
        accept=".csv,text/csv"
        hidden
        onChange={(e) => {
          const f = e.target.files?.[0]
          if (f) handleUpload(f)
        }}
      />
      {file ? (
        <>
          <FileSpreadsheet size={18} className="text-emer shrink-0" />
          <div className="min-w-0 flex-1">
            <div className="mono truncate font-semibold text-indigo-950 text-[11px]">{file}</div>
            <div className="text-[10px] text-slate-500">
              {uploadStatus || (rows !== null ? `${rows} sensor rows` : 'sensor data attached')}
            </div>
          </div>
          <button
            onClick={(e) => {
              e.stopPropagation()
              onFile(null, null)
              setUploadStatus(null)
            }}
            className="text-slate-400 hover:text-[#EC4899]"
          >
            <X size={14} />
          </button>
        </>
      ) : (
        <>
          <Upload size={16} className={uploading ? 'animate-bounce text-royal shrink-0' : 'text-royal shrink-0'} />
          <div className="min-w-0 flex-1 text-[11px]">
            <span className="text-violet-700">
              <b>Drop CSV sensor data</b> or click to upload
            </span>
          </div>
          <button
            onClick={loadSample}
            disabled={uploading}
            className="mono flex items-center gap-1 rounded-lg border border-violet-200 bg-white/90 px-2 py-1 text-[10px] font-semibold text-royal shadow-xs hover:bg-violet-50 shrink-0"
          >
            <Download size={10} /> Load Synthetic CSV
          </button>
        </>
      )}
    </div>
  )
}
