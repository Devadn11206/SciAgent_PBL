import { CheckCircle2 } from 'lucide-react'
import type { AblationRow, BenchRow } from '@/data/types'
import { sci } from '@/data/format'
import { cx } from './ui'

const KIND_DOT: Record<BenchRow['kind'], string> = { reference: '#1E1B4B', baseline: '#F472B6', agent: '#EC4899', fno: '#06B6D4' }

export function BenchTable({ rows, highlight, paper }: { rows: BenchRow[]; highlight?: BenchRow['kind']; paper?: boolean }) {
  const best = (get: (r: BenchRow) => number | null) => {
    const vals = rows.filter((r) => r.kind !== 'reference').map(get).filter((v): v is number => v !== null)
    return Math.min(...vals)
  }
  const cols: { key: string; label: string; get: (r: BenchRow) => number | null; fmt: (v: number | null) => string; lowerBest: boolean }[] = [
    { key: 'l2', label: 'Relative L2', get: (r) => r.relL2, fmt: (v) => sci(v), lowerBest: true },
    { key: 'li', label: 'L∞ Error', get: (r) => r.linf, fmt: (v) => (v === null ? '—' : sci(v)), lowerBest: true },
    { key: 'res', label: paper ? 'Residual' : 'Mean PDE Residual', get: (r) => r.meanRes, fmt: (v) => sci(v), lowerBest: true },
    { key: 'dr', label: paper ? 'Drift' : 'Conservation Drift', get: (r) => r.driftPct, fmt: (v) => `${v}%`, lowerBest: true },
    { key: 'w', label: paper ? 'Time' : 'Wall-Clock', get: (r) => r.wallS, fmt: (v) => `${v}s`, lowerBest: true },
    { key: 'm', label: paper ? 'Mem' : 'Peak Memory', get: (r) => r.memMB, fmt: (v) => `${v} MB`, lowerBest: true },
  ]
  return (
    <div className="overflow-x-auto">
      <table className={cx('w-full border-separate border-spacing-0 text-left', paper ? 'text-[11.5px]' : 'text-[13px]')}>
        <thead>
          <tr>
            <th className="label border-b border-violet-100 px-3 py-2.5">Method</th>
            {cols.map((c) => <th key={c.key} className="label whitespace-nowrap border-b border-violet-100 px-3 py-2.5 text-right">{c.label}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const hl = highlight === r.kind
            return (
              <tr key={r.method} className={cx('transition', hl ? 'bg-pink-50/80' : 'hover:bg-violet-50/40')}>
                <td className="border-b border-violet-50 px-3 py-2.5">
                  <div className="flex items-center gap-2 font-semibold text-indigo-950"><span className="h-2.5 w-2.5 rounded-full" style={{ background: KIND_DOT[r.kind] }} />{r.method}</div>
                  {!paper && <div className="pl-[18px] text-[11px] text-slate-500">{r.detail}</div>}
                </td>
                {cols.map((c) => {
                  const v = c.get(r)
                  const isBest = r.kind !== 'reference' && v !== null && v === best(c.get)
                  const ref = r.kind === 'reference' && (c.key === 'l2' || c.key === 'li')
                  return (
                    <td key={c.key} className={cx('mono border-b border-violet-50 px-3 py-2.5 text-right', isBest ? 'font-bold text-emerald-600' : 'text-indigo-950', ref && 'text-slate-400')}>
                      {ref ? 'ref.' : c.fmt(v)}
                    </td>
                  )
                })}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

export function AblationTable({ rows, paper }: { rows: AblationRow[]; paper?: boolean }) {
  return (
    <div className="overflow-x-auto">
      <table className={cx('w-full border-separate border-spacing-0 text-left', paper ? 'text-[11.5px]' : 'text-[13px]')}>
        <thead>
          <tr>
            {['#', 'Trial', 'Critic action', 'Diagnosis', 'Rel. L2', 'Δ vs previous', 'Verdict'].map((h, i) => (
              <th key={h} className={cx('label border-b border-violet-100 px-3 py-2.5', i >= 4 && 'text-right')}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => {
            const delta = i === 0 ? null : ((r.relL2 - rows[i - 1].relL2) / rows[i - 1].relL2) * 100
            const last = i === rows.length - 1
            return (
              <tr key={i} className={cx(last && 'bg-emerald-50/60')}>
                <td className="mono border-b border-violet-50 px-3 py-2.5 text-slate-400">{i}</td>
                <td className="border-b border-violet-50 px-3 py-2.5"><span className="mono rounded-full bg-violet-100 px-2 py-0.5 text-[11px] font-bold text-royal">{r.trial}</span></td>
                <td className="border-b border-violet-50 px-3 py-2.5 font-semibold text-indigo-950">{r.action}</td>
                <td className="border-b border-violet-50 px-3 py-2.5 text-slate-600">{r.diagnosis}</td>
                <td className="mono whitespace-nowrap border-b border-violet-50 px-3 py-2.5 text-right font-semibold text-indigo-950">{sci(r.relL2)}</td>
                <td className={cx('mono border-b border-violet-50 px-3 py-2.5 text-right font-semibold', delta === null ? 'text-slate-400' : 'text-emerald-600')}>{delta === null ? '—' : `${delta.toFixed(1)}%`}</td>
                <td className="border-b border-violet-50 px-3 py-2.5 text-right">
                  {i === 0 ? <span className="text-xs font-semibold text-slate-400">baseline</span> : last ? <span className="inline-flex items-center gap-1 text-xs font-bold text-emerald-600"><CheckCircle2 size={13} /> final</span> : <span className="text-xs font-semibold text-turq">accepted</span>}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
