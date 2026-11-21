import { useMemo } from 'react'
import { Code2, FileDown, FileText, FileType2, Images } from 'lucide-react'
import type { Experiment } from '@/data/types'
import { buildFields, buildSlices } from '@/data/fieldGenerators'
import { heatFigure, overlayFigure } from '@/data/figures'
import { Equation } from '@/components/Equation'
import { PlotlyChart } from '@/components/PlotlyChart'
import { AblationTable, BenchTable } from '@/components/Tables'
import { sci } from '@/data/format'

export interface ExportActions { pdf: () => void; tex: () => void; figs: () => void; py: () => void; busy: boolean }

export function ReportView({ exp, actions }: { exp: Experiment; actions: ExportActions }) {
  const f = useMemo(() => buildFields(exp), [exp])
  const slices = useMemo(() => buildSlices(exp, f), [exp, f])
  const figA = useMemo(() => heatFigure('ref', exp, f, '2d', 'corrected', false, { height: 260, compact: true }), [exp, f])
  const figB = useMemo(() => heatFigure('pred', exp, f, '2d', 'corrected', false, { height: 260, compact: true }), [exp, f])
  const figC = useMemo(() => heatFigure('err', exp, f, '2d', 'corrected', false, { height: 260, compact: true }), [exp, f])
  const figD = useMemo(() => overlayFigure(exp, f, slices, true, { height: 260, compact: true }), [exp, f, slices])
  const r = exp.report
  const agent = exp.bench.find((b) => b.kind === 'agent')!
  const base = exp.bench.find((b) => b.kind === 'baseline')!
  const date = new Date().toLocaleDateString('en-GB', { day: 'numeric', month: 'long', year: 'numeric' })

  const buttons = [
    { label: 'Download Single-Column PDF (Typst)', icon: FileDown, run: actions.pdf, primary: true },
    { label: 'Copy LaTeX Tables (.tex)', icon: FileType2, run: actions.tex },
    { label: 'Download 300 DPI Figures (.zip)', icon: Images, run: actions.figs },
    { label: 'Export Reproducible Script (.py)', icon: Code2, run: actions.py },
  ]
  const figs = [
    { fig: figA, cap: `Reference solution computed with ${exp.referenceSolver}.` },
    { fig: figB, cap: 'Agent-corrected PINN prediction.' },
    { fig: figC, cap: `Pointwise absolute error; max ${sci(Math.max(...f.err.corrected.flat()))}.` },
    { fig: figD, cap: 'Time-slice overlay with 95% UQ bands.' },
  ]

  return (
    <div className="space-y-5">
      <div className="no-print glass flex flex-wrap items-center justify-center gap-3 !rounded-full px-5 py-3">
        {buttons.map((b) => (
          <button key={b.label} onClick={b.run} disabled={actions.busy} className={b.primary ? 'btn-primary !py-2 text-[13px]' : 'btn-ghost text-[13px]'}>
            <b.icon size={15} /> {b.label}
          </button>
        ))}
      </div>

      <article className="print-sheet mx-auto max-w-[820px] overflow-hidden rounded-3xl bg-white shadow-[0_30px_80px_-20px_rgba(124,58,237,0.35),0_8px_24px_-8px_rgba(236,72,153,0.2)] ring-1 ring-violet-100" style={{ fontFamily: 'Georgia, "Times New Roman", serif' }}>
        <header className="relative overflow-hidden bg-gradient-to-br from-indigo-950 via-[#4C1D95] to-[#BE185D] px-10 py-9 text-white">
          <div className="absolute -right-10 -top-16 h-52 w-52 rounded-full bg-cyan-400/25 blur-3xl" />
          <div className="relative">
            <div className="mono mb-3 text-[10px] font-semibold uppercase tracking-[0.25em] text-cyan-200">SciAgent · Autonomous Research Report · Run {exp.id}</div>
            <h1 className="text-[26px] font-bold leading-tight">{r.title}</h1>
            <p className="mt-2 text-sm italic text-pink-100">{r.subtitle}</p>
            <div className="mono mt-5 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-indigo-200">
              <span>SciAgent Multi-Agent Pipeline (LangGraph)</span><span>·</span><span>Local Ollama</span><span>·</span><span>{date}</span>
            </div>
          </div>
        </header>

        <div className="space-y-7 px-10 py-8 text-[13.5px] leading-[1.7] text-slate-800">
          <div className="rounded-r-2xl border-l-4 border-royal bg-violet-50/70 px-5 py-4">
            <div className="label mb-2 !text-royal" style={{ fontFamily: '"Plus Jakarta Sans", sans-serif' }}>Executive Findings</div>
            <ul className="list-disc space-y-1.5 pl-5">{r.findings.map((x, i) => <li key={i}>{x}</li>)}</ul>
          </div>

          <div className="text-[12px] text-slate-500"><b className="text-slate-700">Keywords:</b> {r.keywords.join(' · ')}</div>

          <section>
            <h2 className="mb-2 text-[17px] font-bold text-indigo-950" style={{ fontFamily: '"Plus Jakarta Sans", sans-serif' }}>1. Governing Nonlinear Equation</h2>
            <p className="mb-1">{r.eqNote}</p>
            <div className="relative my-3 rounded-xl bg-gradient-to-r from-violet-50/50 via-white to-pink-50/50 py-3">
              <Equation latex={exp.latex} className="text-[1.2rem]" />
              <span className="absolute right-4 top-1/2 -translate-y-1/2 text-sm text-slate-400">(1)</span>
            </div>
            <p className="text-[12.5px] text-slate-600">{r.methods}</p>
          </section>

          <section>
            <h2 className="mb-3 text-[17px] font-bold text-indigo-950" style={{ fontFamily: '"Plus Jakarta Sans", sans-serif' }}>2. Results</h2>
            <div className="grid grid-cols-2 gap-3">
              {figs.map((x, i) => (
                <figure key={i} className={`rounded-xl border border-violet-100 bg-white p-2`}>
                  <PlotlyChart data={x.fig.data} layout={x.fig.layout} height={260} config={{ staticPlot: false }} />
                  <figcaption className="mt-1 px-1 text-[11px] leading-snug text-slate-600"><b>Fig. 1({'abcd'[i]}).</b> {x.cap}</figcaption>
                </figure>
              ))}
            </div>
          </section>

          <section>
            <h2 className="mb-2 text-[17px] font-bold text-indigo-950" style={{ fontFamily: '"Plus Jakarta Sans", sans-serif' }}>3. Benchmark & Ablation</h2>
            <p className="mb-2 text-[12px] font-semibold text-slate-600">Table 1. Benchmark comparison against the numerical reference.</p>
            <div style={{ fontFamily: '"Plus Jakarta Sans", sans-serif' }}><BenchTable rows={exp.bench} highlight="agent" paper /></div>
            <p className="mb-2 mt-5 text-[12px] font-semibold text-slate-600">Table 2. Agentic self-correction ablation log (Critic actions applied cumulatively).</p>
            <div style={{ fontFamily: '"Plus Jakarta Sans", sans-serif' }}><AblationTable rows={exp.ablation} paper /></div>
          </section>

          <section>
            <h2 className="mb-2 text-[17px] font-bold text-indigo-950" style={{ fontFamily: '"Plus Jakarta Sans", sans-serif' }}>4. Physical Interpretation</h2>
            <div className="space-y-3">{r.interpretation.map((p, i) => <p key={i} className="text-justify">{p}</p>)}</div>
            <p className="mt-3 rounded-xl bg-emerald-50 px-4 py-2 text-[12.5px] text-emerald-900" style={{ fontFamily: '"Plus Jakarta Sans", sans-serif' }}>
              <b>Verification statement.</b> Metrics are measured against the {exp.referenceSolver} reference: relative L2 {sci(agent.relL2)} (baseline {sci(base.relL2)}), conservation drift {agent.driftPct}%, mean residual {sci(agent.meanRes)}.
            </p>
          </section>

          <footer className="flex items-center gap-2 border-t border-violet-100 pt-4 text-[11px] text-slate-400" style={{ fontFamily: '"Plus Jakarta Sans", sans-serif' }}>
            <FileText size={13} /> Generated by SciAgent · every figure and sentence is derived from the run's verified metrics.
          </footer>
        </div>
      </article>
    </div>
  )
}
