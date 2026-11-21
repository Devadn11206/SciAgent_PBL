import { useEffect, useState } from 'react'
import { BarChart3, ChevronDown, Download, FileCode2, FileText, FlaskConical, History, Images, Network, ScrollText, Sparkles, Table2, Terminal } from 'lucide-react'
import { PRESET_METAS } from '@/data/presetsCatalog'
import { cx, Dropdown } from './ui'
import { fetchExperimentHistory, fetchOllamaStatus, type OllamaStatusResponse } from '@/services/api'
import { MCPInspectorModal } from '@/components/MCPInspectorModal'

export type ViewId = 'studio' | 'orchestrator' | 'benchmark' | 'report'
const TABS: { id: ViewId; label: string; icon: typeof FlaskConical }[] = [
  { id: 'studio', label: 'Problem & Equation Studio', icon: FlaskConical },
  { id: 'orchestrator', label: 'Multi-Agent Orchestrator', icon: Network },
  { id: 'benchmark', label: 'Benchmarking & Plots', icon: BarChart3 },
  { id: 'report', label: 'Single-Column Report', icon: ScrollText },
]
const STATUS = { completed: '#10B981', running: '#7C3AED', failed: '#EC4899', COMPLETED: '#10B981', RUNNING: '#7C3AED', FAILED: '#EC4899' } as const

interface Props {
  view: ViewId
  onView: (v: ViewId) => void
  currentRunId?: string | null
  onPickHistoryRun: (runId: string) => void
  onExport: { pdf: () => void; tex: () => void; figs: () => void; py: () => void }
  ollamaStatus?: OllamaStatusResponse
}

export function NavBar({ view, onView, currentRunId, onPickHistoryRun, onExport, ollamaStatus }: Props) {
  const [ollama, setOllama] = useState<OllamaStatusResponse>(ollamaStatus || { online: false, model: 'checking…', supported_models: [] })
  const [dbHistory, setDbHistory] = useState<any[]>([])
  const [mcpModalOpen, setMcpModalOpen] = useState(false)

  useEffect(() => {
    if (ollamaStatus) {
      setOllama(ollamaStatus)
    }
  }, [ollamaStatus])

  useEffect(() => {
    fetchExperimentHistory().then(setDbHistory).catch(() => {})
    const interval = setInterval(() => {
      fetchExperimentHistory().then(setDbHistory).catch(() => {})
    }, 10000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="no-print fixed inset-x-0 top-3 z-40 flex justify-center px-2 md:px-4">
      <nav className="glass flex w-full max-w-[1600px] flex-nowrap items-center justify-between gap-1.5 md:gap-2.5 !rounded-full px-2.5 md:px-4 py-2">
        <div className="flex shrink-0 items-center gap-2 pl-0.5 pr-1 md:pr-2">
          <span className="flex h-8 w-8 md:h-9 md:w-9 items-center justify-center rounded-2xl bg-gradient-to-br from-royal via-fuchsia-500 to-turq text-white shadow-lg shadow-violet-300/60"><Sparkles size={17} /></span>
          <span className="text-[16px] md:text-[17px] font-extrabold tracking-tight text-indigo-950">Sci<span className="text-royal">Agent</span></span>
        </div>

        <div className="flex flex-nowrap items-center gap-1 md:gap-1.5">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => onView(t.id)}
              className={cx(
                'flex shrink-0 items-center gap-1.5 whitespace-nowrap rounded-full px-2.5 py-1.5 text-xs xl:px-3.5 xl:py-2 xl:text-[13px] font-semibold transition',
                view === t.id ? 'bg-royal text-white shadow-md shadow-violet-300/60' : 'text-violet-800 hover:bg-violet-100/70'
              )}
            >
              <t.icon size={14} /> {t.label}
            </button>
          ))}
        </div>

        <div className="flex shrink-0 flex-nowrap items-center gap-1.5 md:gap-2">
          <Dropdown
            trigger={(open, toggle) => (
              <button onClick={toggle} className="pill flex items-center gap-1.5 px-3 py-2 text-[13px] font-semibold text-violet-800 hover:bg-white">
                <History size={14} /> <span className="hidden 2xl:inline">Experiment History</span><span className="2xl:hidden">History</span>
                <ChevronDown size={13} className={cx('transition', open && 'rotate-180')} />
              </button>
            )}
          >
            {(close) => (
              <div className="w-[360px] max-h-[400px] overflow-y-auto">
                <div className="label px-3 py-2">SQLite Experiment Records ({dbHistory.length})</div>
                {dbHistory.length === 0 ? (
                  <div className="p-4 text-center text-xs text-slate-400">No SQLite experiment records yet. Run an experiment to see it recorded here.</div>
                ) : (
                  dbHistory.map((h) => {
                    const meta = PRESET_METAS[h.problem_id] || { title: h.problem_id.replace('_', ' ').toUpperCase() }
                    const relL2 = h.final_metrics?.rel_l2 ? h.final_metrics.rel_l2.toExponential(2) : '—'
                    const statusColor = (STATUS as any)[h.status] || '#10B981'
                    const formattedDate = h.created_at ? (() => {
                      try {
                        const d = new Date(h.created_at.includes('T') ? h.created_at : h.created_at.replace(' ', 'T'))
                        return isNaN(d.getTime()) ? h.created_at : d.toLocaleString([], {
                          year: 'numeric',
                          month: 'short',
                          day: 'numeric',
                          hour: '2-digit',
                          minute: '2-digit',
                          second: '2-digit',
                        })
                      } catch {
                        return h.created_at
                      }
                    })() : 'Just now'
                    return (
                      <button
                        key={h.id}
                        onClick={() => { onPickHistoryRun(h.id); close() }}
                        className={cx(
                          'flex w-full items-start gap-3 rounded-xl px-3 py-2 text-left transition hover:bg-violet-50',
                          h.id === currentRunId && 'bg-violet-50/70'
                        )}
                      >
                        <span className="mt-1.5 h-2 w-2 shrink-0 rounded-full" style={{ background: statusColor }} />
                        <span className="min-w-0 flex-1">
                          <span className="block truncate text-[13px] font-semibold text-indigo-950">{meta.title}</span>
                          <span className="block truncate text-[11px] text-slate-500">{formattedDate} · {h.total_trials} trial(s)</span>
                        </span>
                        <span className="mono text-[11px] font-semibold text-slate-600">L2 {relL2}</span>
                      </button>
                    )
                  })
                )}
              </div>
            )}
          </Dropdown>

          {/* Advanced FastMCP Server Pill */}
          <button
            onClick={() => setMcpModalOpen(true)}
            className="flex items-center gap-1.5 rounded-full border border-violet-200 bg-violet-50/70 px-3 py-2 text-[13px] font-semibold text-royal transition hover:bg-violet-100 hover:border-violet-300"
            title="Inspect Advanced FastMCP Core Tools & Client Config"
          >
            <Terminal size={14} />
            <span className="hidden xl:inline">MCP v2: 5 Tools · Streamable</span>
            <span className="xl:hidden">MCP</span>
          </button>

          <span className={cx(
            'flex items-center gap-2 rounded-full border px-3 py-2 text-[13px] font-semibold transition',
            ollama.online ? 'border-emerald-200 bg-emerald-50 text-emerald-700' : 'border-amber-200 bg-amber-50 text-amber-700'
          )}>
            <span className="relative flex h-2 w-2">
              <span className={cx('absolute inline-flex h-full w-full rounded-full opacity-60', ollama.online ? 'animate-ping bg-emerald-400' : 'bg-amber-400')} />
              <span className={cx('relative inline-flex h-2 w-2 rounded-full', ollama.online ? 'bg-emerald-500' : 'bg-amber-500')} />
            </span>
            <span className="hidden xl:inline">
              Ollama: {ollama.online ? ollama.model : 'Deterministic Offline Mode'}
            </span>
            <span className="xl:hidden">Ollama</span>
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
                  { label: 'Single-column PDF (Typst)', icon: FileText, run: onExport.pdf },
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

      {/* MCP Inspector Modal */}
      <MCPInspectorModal isOpen={mcpModalOpen} onClose={() => setMcpModalOpen(false)} />
    </div>
  )
}
