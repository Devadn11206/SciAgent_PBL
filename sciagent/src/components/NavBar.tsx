import { BarChart3, ChevronDown, Download, FileCode2, FileText, FlaskConical, History, Images, Network, ScrollText, Sparkles, Table2 } from 'lucide-react'
import { experiments, history } from '@/data/mockExperiments'
import type { ExperimentId } from '@/data/types'
import { cx, Dropdown } from './ui'

export type ViewId = 'studio' | 'orchestrator' | 'benchmark' | 'report'
const TABS: { id: ViewId; label: string; icon: typeof FlaskConical }[] = [
  { id: 'studio', label: 'Problem & Equation Studio', icon: FlaskConical },
  { id: 'orchestrator', label: 'Multi-Agent Orchestrator', icon: Network },
  { id: 'benchmark', label: 'Benchmarking & Plots', icon: BarChart3 },
  { id: 'report', label: 'Single-Column Report', icon: ScrollText },
]
const STATUS = { verified: '#10B981', 'self-corrected': '#7C3AED', failed: '#EC4899' } as const

interface Props {
  view: ViewId
  onView: (v: ViewId) => void
  expId: ExperimentId
  onPickRun: (id: ExperimentId) => void
  onExport: { pdf: () => void; tex: () => void; figs: () => void; py: () => void }
}

export function NavBar({ view, onView, expId, onPickRun, onExport }: Props) {
  return (
    <div className="no-print fixed inset-x-0 top-3 z-40 flex justify-center px-3">
      <nav className="glass flex w-full max-w-[1560px] items-center gap-2 !rounded-full px-3 py-2">
        <div className="flex shrink-0 items-center gap-2 pl-1 pr-2">
          <span className="flex h-9 w-9 items-center justify-center rounded-2xl bg-gradient-to-br from-royal via-fuchsia-500 to-turq text-white shadow-lg shadow-violet-300/60"><Sparkles size={18} /></span>
          <span className="text-[17px] font-extrabold tracking-tight text-indigo-950">Sci<span className="text-royal">Agent</span></span>
        </div>

        <div className="mx-auto flex items-center gap-1 overflow-x-auto">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => onView(t.id)}
              className={cx('flex shrink-0 items-center gap-1.5 whitespace-nowrap rounded-full px-3.5 py-2 text-[13px] font-semibold transition', view === t.id ? 'bg-royal text-white shadow-md shadow-violet-300/60' : 'text-violet-800 hover:bg-violet-100/70')}
            >
              <t.icon size={14} /> {t.label}
            </button>
          ))}
        </div>

        <div className="flex shrink-0 items-center gap-2">
          <Dropdown
            trigger={(open, toggle) => (
              <button onClick={toggle} className="pill flex items-center gap-1.5 px-3 py-2 text-[13px] font-semibold text-violet-800 hover:bg-white">
                <History size={14} /> <span className="hidden 2xl:inline">Experiment History</span><span className="2xl:hidden">History</span>
                <ChevronDown size={13} className={cx('transition', open && 'rotate-180')} />
              </button>
            )}
          >
            {(close) => (
              <div className="w-[340px]">
                <div className="label px-3 py-2">Recent experiments</div>
                {history.map((h) => {
                  const e = experiments.find((x) => x.id === h.expId)!
                  return (
                    <button key={h.id} onClick={() => { onPickRun(h.expId); close() }} className={cx('flex w-full items-start gap-3 rounded-xl px-3 py-2 text-left transition hover:bg-violet-50', h.expId === expId && 'bg-violet-50/70')}>
                      <span className="mt-1.5 h-2 w-2 shrink-0 rounded-full" style={{ background: STATUS[h.status] }} />
                      <span className="min-w-0 flex-1">
                        <span className="block truncate text-[13px] font-semibold text-indigo-950">{e.title}</span>
                        <span className="block truncate text-[11px] text-slate-500">{h.when} · {h.note}</span>
                      </span>
                      <span className="mono text-[11px] font-semibold text-slate-600">{h.relL2}</span>
                    </button>
                  )
                })}
              </div>
            )}
          </Dropdown>

          <span className="flex items-center gap-2 rounded-full border border-emerald-200 bg-emerald-50 px-3 py-2 text-[13px] font-semibold text-emerald-700">
            <span className="relative flex h-2 w-2"><span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-60" /><span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-500" /></span>
            <span className="hidden xl:inline">Local Ollama: Connected</span><span className="xl:hidden">Ollama</span>
          </span>

          <Dropdown
            trigger={(_, toggle) => (
              <button onClick={toggle} className="flex items-center gap-1.5 rounded-full bg-indigo-950 px-3.5 py-2 text-[13px] font-semibold text-white transition hover:bg-indigo-900" aria-label="Quick export">
                <Download size={14} /> <span className="hidden xl:inline">Export</span>
              </button>
            )}
          >
            {(close) => (
              <div className="w-[270px]">
                {[
                  { label: 'Single-column PDF (print)', icon: FileText, run: onExport.pdf },
                  { label: 'LaTeX tables (.tex)', icon: Table2, run: onExport.tex },
                  { label: 'Figures 300 DPI (.zip)', icon: Images, run: onExport.figs },
                  { label: 'Reproducible script (.py)', icon: FileCode2, run: onExport.py },
                ].map((i) => (
                  <button key={i.label} onClick={() => { i.run(); close() }} className="flex w-full items-center gap-2.5 rounded-xl px-3 py-2 text-[13px] font-semibold text-indigo-950 transition hover:bg-violet-50">
                    <i.icon size={15} className="text-royal" /> {i.label}
                  </button>
                ))}
              </div>
            )}
          </Dropdown>
        </div>
      </nav>
    </div>
  )
}
