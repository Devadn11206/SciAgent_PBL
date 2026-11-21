import { useEffect, useState } from 'react'
import {
  Code2,
  ExternalLink,
  FileDown,
  FileType2,
  Images,
  Loader2,
  RefreshCw,
  Sliders,
  ChevronDown,
  ChevronUp,
  AlertCircle,
  Copy,
  Check,
} from 'lucide-react'
import type { ProblemIR } from '@/data/types'
import { PRESET_METAS } from '@/data/presetsCatalog'
import { Equation } from '@/components/Equation'
import { CriticChatDrawer } from '@/components/CriticChatDrawer'
import {
  generateReportBundle,
  getReportPdfUrl,
  recompileReport,
  type AgentRunResponse,
} from '@/services/api'

export interface ExportActions {
  pdf: () => void
  tex: () => void
  figs: () => void
  py: () => void
  busy: boolean
}

interface Props {
  activeProblem: ProblemIR
  activeRunId: string | null
  agentResult: AgentRunResponse | null
  actions: ExportActions
}

export function ReportView({ activeProblem, activeRunId, agentResult, actions }: Props) {
  const [pdfReady, setPdfReady] = useState(false)
  const [pdfLoading, setPdfLoading] = useState(false)
  const [pdfUrl, setPdfUrl] = useState<string | null>(null)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  // 3-tab view switcher: 'pdf' | 'typst' | 'latex'
  const [viewTab, setViewTab] = useState<'pdf' | 'typst' | 'latex'>('pdf')
  const [typstSource, setTypstSource] = useState<string>('')
  const [latexTables, setLatexTables] = useState<string>('')
  const [copiedTex, setCopiedTex] = useState(false)
  const [copiedTypst, setCopiedTypst] = useState(false)

  // Collapsible customizer state
  const [showCustomizer, setShowCustomizer] = useState(false)
  const [recompiling, setRecompiling] = useState(false)
  const [customTitle, setCustomTitle] = useState('')
  const [customAuthors, setCustomAuthors] = useState('SciAgent Autonomous Discovery Team')
  const [customAffiliation, setCustomAffiliation] = useState('Laboratory for AI-Accelerated Physical Sciences')
  const [customJournal, setCustomJournal] = useState('Physical Review Letters (PRL)')
  const [customNotes, setCustomNotes] = useState('')

  // Determine effective run ID strictly: do NOT fake an ID if there is genuinely no run
  const hasRealRun = Boolean(activeRunId || agentResult?.run_id)
  const effectiveRunId = activeRunId || agentResult?.run_id || null

  const meta = PRESET_METAS[activeProblem.id] || {
    id: activeProblem.id,
    title: activeProblem.name,
    subtitle: 'Autonomous Scientific Discovery Report',
    referenceSolver: 'High-Order Numerical Solver',
  }

  // Set default title when problem changes
  useEffect(() => {
    if (!customTitle) {
      setCustomTitle(`Autonomous Scientific Discovery & Neural Verification: ${activeProblem.name}`)
    }
  }, [activeProblem.name, customTitle])

  // Trigger PDF generation when a real run is active
  useEffect(() => {
    if (!effectiveRunId) {
      setPdfLoading(false)
      setPdfReady(false)
      setPdfUrl(null)
      setErrorMsg(null)
      return
    }

    let active = true
    setPdfLoading(true)
    setErrorMsg(null)

    generateReportBundle(effectiveRunId)
      .then((res: any) => {
        if (active) {
          setPdfUrl(`${getReportPdfUrl(effectiveRunId)}?t=${Date.now()}`)
          setPdfReady(true)
          if (res?.typst_source) setTypstSource(res.typst_source)
        }
      })
      .catch((e) => {
        console.warn('PDF pre-compilation notice:', e)
        if (active) {
          // Attempt direct fallback check
          setPdfUrl(`${getReportPdfUrl(effectiveRunId)}?t=${Date.now()}`)
          setPdfReady(true)
        }
      })
      .finally(() => {
        if (active) setPdfLoading(false)
      })

    return () => {
      active = false
    }
  }, [effectiveRunId])

  // Recompile handler
  const handleRecompile = async () => {
    if (!effectiveRunId) return
    setRecompiling(true)
    setErrorMsg(null)
    try {
      const res = await recompileReport(effectiveRunId, {
        title: customTitle,
        authors: customAuthors,
        affiliation: customAffiliation,
        journal: customJournal,
        custom_notes: customNotes,
        typst_override: viewTab === 'typst' ? typstSource : undefined,
      })
      setPdfUrl(`${getReportPdfUrl(effectiveRunId)}?t=${Date.now()}`)
      setPdfReady(true)
      if (res.typst_source) setTypstSource(res.typst_source)
      if (res.latex_tables) setLatexTables(res.latex_tables)
    } catch (err: any) {
      setErrorMsg(`Recompilation failed: ${err.message || 'Unknown error'}`)
    } finally {
      setRecompiling(false)
    }
  }

  const handleCopyTex = () => {
    if (!latexTables) return
    navigator.clipboard.writeText(latexTables)
    setCopiedTex(true)
    setTimeout(() => setCopiedTex(false), 2000)
  }

  const handleCopyTypst = () => {
    if (!typstSource) return
    navigator.clipboard.writeText(typstSource)
    setCopiedTypst(true)
    setTimeout(() => setCopiedTypst(false), 2000)
  }

  const buttons = [
    { label: 'Download Single-Column PDF (Typst)', icon: FileDown, run: actions.pdf, primary: true },
    { label: 'Copy LaTeX Tables (.tex)', icon: FileType2, run: actions.tex },
    { label: 'Download 300 DPI Figures (.zip)', icon: Images, run: actions.figs },
    { label: 'Export Reproducible Script (.py)', icon: Code2, run: actions.py },
  ]

  const dbRec = agentResult?.db_record
  const metrics = dbRec?.final_metrics
  const relL2Str = metrics?.rel_l2 ? metrics.rel_l2.toExponential(2) : '—'
  const driftStr = metrics?.drift_pct !== undefined ? `${metrics.drift_pct.toFixed(2)}%` : '—'
  const trialsCount = dbRec?.total_trials || agentResult?.total_trials || 1

  // Clean empty state if user hasn't run anything yet
  if (!hasRealRun || !effectiveRunId) {
    return (
      <div className="space-y-6">
        <div className="mx-auto max-w-[860px] rounded-3xl border border-violet-100 bg-white/90 p-12 text-center shadow-xl backdrop-blur-md">
          <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-violet-100 text-royal shadow-sm">
            <FileDown size={32} />
          </div>
          <h2 className="text-xl font-bold text-slate-800 mb-2">No Active Experiment Report Yet</h2>
          <p className="mx-auto max-w-lg text-sm text-slate-600 mb-6">
            Configure an equation in <strong>Problem & Equation Studio</strong> and click{' '}
            <strong>"Run Autonomous Agent Workflow"</strong>, or pick a completed run from{' '}
            <strong>Experiment History</strong> to compile its publication-ready single-column Typst manuscript.
          </p>
          <div className="inline-flex items-center gap-2 rounded-xl bg-violet-50 px-4 py-2 text-xs font-semibold text-royal border border-violet-200">
            <span>Awaiting live execution to synthesize 300 DPI figures & Typst 0.15 manuscript</span>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Top Action Bar */}
      <div className="no-print glass flex flex-wrap items-center justify-center gap-3 !rounded-full px-5 py-3">
        {buttons.map((b) => (
          <button
            key={b.label}
            onClick={b.run}
            disabled={actions.busy}
            className={b.primary ? 'btn-primary !py-2 text-[13px]' : 'btn-ghost text-[13px]'}
          >
            <b.icon size={15} /> {b.label}
          </button>
        ))}
      </div>

      {/* Collapsible Manuscript Customizer Panel */}
      <div className="mx-auto max-w-[860px] rounded-2xl border border-violet-200 bg-white/95 p-5 shadow-lg backdrop-blur-sm">
        <div className="flex items-center justify-between">
          <button
            onClick={() => setShowCustomizer(!showCustomizer)}
            className="flex items-center gap-2 text-sm font-bold text-slate-800 hover:text-royal transition-colors"
          >
            <Sliders size={16} className="text-royal" />
            <span>Customize Paper Metadata & Discussion Notes</span>
            {showCustomizer ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </button>
          <button
            onClick={handleRecompile}
            disabled={recompiling || pdfLoading}
            className="btn-primary !py-1.5 !px-3 text-xs flex items-center gap-1.5"
          >
            <RefreshCw size={13} className={recompiling ? 'animate-spin' : ''} />
            {recompiling ? 'Recompiling...' : 'Recompile PDF'}
          </button>
        </div>

        {showCustomizer && (
          <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4 border-t border-slate-100 pt-4 text-xs">
            <div>
              <label className="block font-semibold text-slate-700 mb-1">Paper Title</label>
              <input
                type="text"
                value={customTitle}
                onChange={(e) => setCustomTitle(e.target.value)}
                className="w-full rounded-lg border border-slate-200 p-2 text-xs focus:border-royal focus:outline-none"
                placeholder="Manuscript Title..."
              />
            </div>
            <div>
              <label className="block font-semibold text-slate-700 mb-1">Target Journal / Venue</label>
              <input
                type="text"
                value={customJournal}
                onChange={(e) => setCustomJournal(e.target.value)}
                className="w-full rounded-lg border border-slate-200 p-2 text-xs focus:border-royal focus:outline-none"
                placeholder="Physical Review Letters (PRL)"
              />
            </div>
            <div>
              <label className="block font-semibold text-slate-700 mb-1">Authors</label>
              <input
                type="text"
                value={customAuthors}
                onChange={(e) => setCustomAuthors(e.target.value)}
                className="w-full rounded-lg border border-slate-200 p-2 text-xs focus:border-royal focus:outline-none"
                placeholder="SciAgent Autonomous Discovery Team"
              />
            </div>
            <div>
              <label className="block font-semibold text-slate-700 mb-1">Affiliation</label>
              <input
                type="text"
                value={customAffiliation}
                onChange={(e) => setCustomAffiliation(e.target.value)}
                className="w-full rounded-lg border border-slate-200 p-2 text-xs focus:border-royal focus:outline-none"
                placeholder="Laboratory for AI-Accelerated Physical Sciences"
              />
            </div>
            <div className="md:col-span-2">
              <label className="block font-semibold text-slate-700 mb-1">
                Custom Discussion & Physical Interpretation Notes (Appends to Section 5)
              </label>
              <textarea
                rows={3}
                value={customNotes}
                onChange={(e) => setCustomNotes(e.target.value)}
                className="w-full rounded-lg border border-slate-200 p-2 text-xs focus:border-royal focus:outline-none font-mono"
                placeholder="Add observational notes, physical interpretations, experimental bounds, or caveats to compile directly into the paper..."
              />
            </div>
          </div>
        )}
      </div>

      {/* Error banner if recompile fails */}
      {errorMsg && (
        <div className="mx-auto max-w-[860px] rounded-xl border border-red-200 bg-red-50 p-4 text-xs text-red-700 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AlertCircle size={16} />
            <span>{errorMsg}</span>
          </div>
          <button onClick={handleRecompile} className="font-semibold underline hover:text-red-900">
            Retry
          </button>
        </div>
      )}

      {/* Document View Switcher Header */}
      <section className="no-print mx-auto max-w-[860px] rounded-2xl border border-violet-200 bg-white p-5 shadow-xl">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 pb-3">
          <div className="flex items-center gap-2">
            <span className="flex h-2.5 w-2.5 rounded-full bg-emerald-500 animate-pulse" />
            <h3 className="text-sm font-bold tracking-tight text-slate-800">
              Single-Column Academic Report (Typst 0.15 Engine)
            </h3>
          </div>

          <div className="flex items-center gap-3">
            {/* View Switcher Tabs */}
            <div className="flex rounded-lg bg-slate-100 p-1 text-xs font-semibold">
              <button
                onClick={() => setViewTab('pdf')}
                className={`rounded-md px-3 py-1 transition-all ${
                  viewTab === 'pdf' ? 'bg-white text-royal shadow-sm' : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Rendered PDF
              </button>
              <button
                onClick={() => setViewTab('typst')}
                className={`rounded-md px-3 py-1 transition-all ${
                  viewTab === 'typst' ? 'bg-white text-royal shadow-sm' : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Typst Source (.typ)
              </button>
              <button
                onClick={() => setViewTab('latex')}
                className={`rounded-md px-3 py-1 transition-all ${
                  viewTab === 'latex' ? 'bg-white text-royal shadow-sm' : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                LaTeX Tables (.tex)
              </button>
            </div>

            {pdfUrl && viewTab === 'pdf' && (
              <a
                href={pdfUrl}
                target="_blank"
                rel="noreferrer"
                className="flex items-center gap-1.5 text-xs font-semibold text-royal hover:underline"
              >
                Open in Tab <ExternalLink size={13} />
              </a>
            )}
          </div>
        </div>

        {/* Tab 1: Rendered PDF */}
        {viewTab === 'pdf' && (
          <div>
            {pdfLoading ? (
              <div className="flex h-96 flex-col items-center justify-center gap-3 rounded-xl bg-slate-50 text-slate-500">
                <Loader2 className="animate-spin text-royal" size={32} />
                <p className="text-sm font-medium">Compiling single-column academic Typst document & 300 DPI figures...</p>
              </div>
            ) : pdfReady && pdfUrl ? (
              <iframe
                src={pdfUrl}
                title="Typst Academic PDF"
                className="h-[750px] w-full rounded-xl border border-slate-200 shadow-inner"
              />
            ) : (
              <div className="rounded-xl bg-amber-50 p-4 text-xs text-amber-800 flex items-center justify-between">
                <span>PDF not yet compiled. Click Recompile PDF to generate.</span>
                <button onClick={handleRecompile} className="font-semibold underline">
                  Compile Now
                </button>
              </div>
            )}
          </div>
        )}

        {/* Tab 2: Typst Source Code (.typ) */}
        {viewTab === 'typst' && (
          <div className="relative">
            <div className="mb-2 flex items-center justify-between text-xs text-slate-500">
              <span>Single-Column Typst 0.15 Source code (`report.typ`)</span>
              <button
                onClick={handleCopyTypst}
                className="flex items-center gap-1 rounded bg-slate-100 px-2 py-1 font-semibold hover:bg-slate-200"
              >
                {copiedTypst ? <Check size={12} className="text-emerald-600" /> : <Copy size={12} />}
                {copiedTypst ? 'Copied' : 'Copy .typ Source'}
              </button>
            </div>
            <textarea
              value={typstSource}
              onChange={(e) => setTypstSource(e.target.value)}
              rows={24}
              className="w-full rounded-xl border border-slate-200 bg-slate-900 p-4 font-mono text-xs text-emerald-400 focus:outline-none"
              placeholder="# Typst source will appear here after compilation..."
            />
            <div className="mt-2 flex justify-end">
              <button
                onClick={handleRecompile}
                disabled={recompiling}
                className="btn-primary !py-1.5 !px-3 text-xs flex items-center gap-1.5"
              >
                <RefreshCw size={12} className={recompiling ? 'animate-spin' : ''} />
                Apply Edits & Recompile PDF
              </button>
            </div>
          </div>
        )}

        {/* Tab 3: LaTeX Tables (.tex) */}
        {viewTab === 'latex' && (
          <div className="relative">
            <div className="mb-2 flex items-center justify-between text-xs text-slate-500">
              <span>Publication-grade LaTeX Scorecard & Ablation Tables (`manuscript_tables.tex`)</span>
              <button
                onClick={handleCopyTex}
                className="flex items-center gap-1 rounded bg-slate-100 px-2 py-1 font-semibold hover:bg-slate-200"
              >
                {copiedTex ? <Check size={12} className="text-emerald-600" /> : <Copy size={12} />}
                {copiedTex ? 'Copied' : 'Copy LaTeX'}
              </button>
            </div>
            <textarea
              readOnly
              value={latexTables || '% Click Recompile PDF to extract LaTeX tables...'}
              rows={24}
              className="w-full rounded-xl border border-slate-200 bg-slate-900 p-4 font-mono text-xs text-amber-300 focus:outline-none"
            />
          </div>
        )}
      </section>

      {/* Quick Summary Card */}
      <article className="print-sheet mx-auto max-w-[820px] overflow-hidden rounded-3xl bg-white shadow-xl ring-1 ring-violet-100 p-8">
        <h2 className="text-xl font-bold text-indigo-950 mb-2">{meta.title}</h2>
        <p className="text-sm text-slate-600 mb-4">{meta.subtitle}</p>
        <div className="rounded-xl bg-violet-50/70 p-4 border-l-4 border-royal mb-5">
          <div className="font-semibold text-royal text-xs uppercase mb-1">Executive Summary</div>
          <p className="text-sm text-slate-800">
            {dbRec?.insight_summary ||
              `Autonomous self-correction reduced relative L2 error to ${relL2Str} across ${trialsCount} trials with energy drift of ${driftStr}. Verified against ${meta.referenceSolver}.`}
          </p>
        </div>
        <div className="my-3 py-2 bg-slate-50 rounded-xl px-4 text-center">
          <Equation latex={activeProblem.equation_latex} className="text-lg" />
        </div>
      </article>

      {/* Post-Run "Ask the Critic" Chat Drawer */}
      <div className="mx-auto max-w-[860px]">
        <CriticChatDrawer runId={effectiveRunId} problemName={activeProblem.name} />
      </div>
    </div>
  )
}
