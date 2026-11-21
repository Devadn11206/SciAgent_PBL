import { useCallback, useRef, useState } from 'react'
import { CheckCircle2 } from 'lucide-react'
import { experimentById } from '@/data/mockExperiments'
import type { ExperimentId } from '@/data/types'
import { downloadFigures, downloadText, latexTables, reproScript } from '@/data/exporters'
import { NavBar, type ViewId } from '@/components/NavBar'
import { StudioView } from '@/views/StudioView'
import { OrchestratorView } from '@/views/OrchestratorView'
import { BenchmarkView } from '@/views/BenchmarkView'
import { ReportView } from '@/views/ReportView'

export default function App() {
  const [view, setView] = useState<ViewId>('studio')
  const [expId, setExpId] = useState<ExperimentId>('burgers')
  const [runKey, setRunKey] = useState(0)
  const [toast, setToast] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const finished = useRef(new Set<string>())
  const timer = useRef<number>()
  const exp = experimentById(expId)

  const say = useCallback((m: string) => {
    setToast(m)
    window.clearTimeout(timer.current)
    timer.current = window.setTimeout(() => setToast(null), 2800)
  }, [])

  const actions = {
    busy,
    pdf: () => { say('Opening print dialog — choose “Save as PDF”'); setView('report'); setTimeout(() => window.print(), 400) },
    tex: async () => {
      const tex = latexTables(exp)
      try { await navigator.clipboard.writeText(tex); say('LaTeX tables copied to clipboard') }
      catch { downloadText(`sciagent_${exp.id}_tables.tex`, tex); say('Clipboard blocked — downloaded .tex instead') }
    },
    figs: async () => {
      setBusy(true); say('Rendering 4 figures at 300 DPI…')
      try { await downloadFigures(exp); say('Figures exported (.zip)') } catch { say('Figure export failed') } finally { setBusy(false) }
    },
    py: () => { downloadText(`sciagent_${exp.id}.py`, reproScript(exp), 'text/x-python'); say('Reproducible script downloaded') },
  }

  const select = (id: ExperimentId) => { setExpId(id); say(`Loaded preset: ${experimentById(id).title}`) }
  const run = () => { setRunKey((k) => k + 1); setView('orchestrator') }

  return (
    <div className="min-h-screen">
      <NavBar view={view} onView={setView} expId={expId} onPickRun={(id) => { setExpId(id); setView('orchestrator'); say(`Opened run: ${experimentById(id).title}`) }} onExport={actions} />
      <main className="mx-auto max-w-[1360px] px-4 pb-16 pt-24 md:px-6">
        {view === 'studio' && <StudioView exp={exp} onSelect={select} onRun={run} />}
        {view === 'orchestrator' && <OrchestratorView exp={exp} runKey={runKey} finished={finished.current} />}
        {view === 'benchmark' && <BenchmarkView exp={exp} />}
        {view === 'report' && <ReportView exp={exp} actions={actions} />}
      </main>
      {toast && (
        <div className="no-print fixed bottom-6 left-1/2 z-50 flex -translate-x-1/2 items-center gap-2 rounded-full bg-indigo-950 px-5 py-2.5 text-sm font-semibold text-white shadow-2xl">
          <CheckCircle2 size={16} className="text-emerald-400" /> {toast}
        </div>
      )}
    </div>
  )
}
