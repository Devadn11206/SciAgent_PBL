import { useMemo, useState } from 'react'
import { GitCompare, Grid3x3, Table2, TrendingDown } from 'lucide-react'
import type { Experiment } from '@/data/types'
import { buildFields, buildSlices } from '@/data/fieldGenerators'
import { heatFigure, overlayFigure } from '@/data/figures'
import { PlotlyChart } from '@/components/PlotlyChart'
import { AblationTable, BenchTable } from '@/components/Tables'
import { Card, Checkbox, Segmented } from '@/components/ui'
import { sci } from '@/data/format'

export function BenchmarkView({ exp }: { exp: Experiment }) {
  const [variant, setVariant] = useState<'naive' | 'corrected'>('corrected')
  const [mode, setMode] = useState<'2d' | '3d'>('2d')
  const [uq, setUq] = useState(true)

  const fields = useMemo(() => buildFields(exp), [exp])
  const slices = useMemo(() => buildSlices(exp, fields), [exp, fields])
  const ref = useMemo(() => heatFigure('ref', exp, fields, mode, 'corrected'), [exp, fields, mode])
  const pred = useMemo(() => heatFigure('pred', exp, fields, mode, variant, uq), [exp, fields, mode, variant, uq])
  const err = useMemo(() => heatFigure('err', exp, fields, mode, variant), [exp, fields, mode, variant])
  const overlay = useMemo(() => overlayFigure(exp, fields, slices, uq, { height: 400 }), [exp, fields, slices, uq])

  const base = exp.bench.find((b) => b.kind === 'baseline')!
  const agent = exp.bench.find((b) => b.kind === 'agent')!
  const predTitle = variant === 'naive' ? 'Naive PINN Prediction (Trial 1)' : 'Agent-Corrected PINN Prediction'
  const L2 = variant === 'naive' ? base.relL2 : agent.relL2
  const h = mode === '3d' ? 340 : 310

  return (
    <div className="space-y-5">
      <div className="glass flex flex-wrap items-center justify-between gap-4 !rounded-full px-5 py-3">
        <div className="flex items-center gap-3">
          <GitCompare size={17} className="text-royal" />
          <Segmented
            value={variant}
            onChange={setVariant}
            options={[
              { value: 'naive', label: 'Naive PINN (Trial 1)' },
              { value: 'corrected', label: 'Agent Self-Corrected PINN (Final)' },
            ]}
          />
        </div>
        <div className="flex items-center gap-4">
          <Segmented value={mode} onChange={setMode} options={[{ value: '2d', label: '2D Heatmaps' }, { value: '3d', label: '3D Interactive Surfaces' }]} />
          <Checkbox checked={uq} onChange={setUq} label="Show 95% UQ Confidence Bands" />
        </div>
      </div>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
        <Card title={`Reference Numerical Solution`} icon={<Grid3x3 size={16} className="text-turq" />} right={<span className="mono text-[11px] text-slate-400">{exp.referenceSolver}</span>}>
          <PlotlyChart data={ref.data} layout={ref.layout} height={h} />
        </Card>
        <Card title={predTitle} icon={<Grid3x3 size={16} className="text-rosepink" />} right={<span className="mono rounded-full bg-pink-50 px-2.5 py-0.5 text-[11px] font-semibold text-pink-700">Rel L2 {sci(L2)}</span>}>
          <PlotlyChart data={pred.data} layout={pred.layout} height={h} />
        </Card>
        <Card title="Pointwise Absolute Error Map  |u_pred − u_ref|" icon={<TrendingDown size={16} className="text-royal" />} right={<span className="mono text-[11px] text-slate-400">shared colour scale</span>}>
          <PlotlyChart data={err.data} layout={err.layout} height={400} />
        </Card>
        <Card title="Multi-Method Time-Slice Overlay" icon={<GitCompare size={16} className="text-turq" />} right={<span className="mono text-[11px] text-slate-400">{exp.fieldSpec.sliceVar} slices</span>}>
          <PlotlyChart data={overlay.data} layout={overlay.layout} height={400} />
        </Card>
      </div>

      <Card title="Quantitative Benchmark" icon={<Table2 size={16} className="text-royal" />} right={<span className="text-[11px] font-semibold text-emerald-600">green = best among learned methods</span>}>
        <BenchTable rows={exp.bench} highlight={variant === 'naive' ? 'baseline' : 'agent'} />
      </Card>

      <Card title="Agentic Self-Correction Ablation Log" icon={<TrendingDown size={16} className="text-rosepink" />}>
        <AblationTable rows={exp.ablation} />
      </Card>
    </div>
  )
}
