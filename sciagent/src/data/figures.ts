import type { Experiment } from './types'
import type { Fields, Slice } from './fieldGenerators'
import { FONT } from '@/components/PlotlyChart'

export const SOL_SCALE: [number, string][] = [[0, '#06B6D4'], [0.5, '#FFF5F7'], [1, '#EC4899']]
export const PRED_SCALE: [number, string][] = [[0, '#0891B2'], [0.5, '#FDF2F8'], [1, '#DB2777']]
export const ERR_SCALE: [number, string][] = [[0, '#FFFFFF'], [0.35, '#FBCFE8'], [0.7, '#C084FC'], [1, '#6D28D9']]

export const METHOD_COLORS = {
  ref: '#1E1B4B',
  baseline: '#F472B6',
  agent: '#EC4899',
  fno: '#06B6D4',
}

const flat = (z: number[][]) => z.flat()
const maxOf = (z: number[][]) => Math.max(...flat(z))
const minOf = (z: number[][]) => Math.min(...flat(z))

export type PanelKind = 'ref' | 'pred' | 'err'

export function heatFigure(
  kind: PanelKind,
  exp: Experiment,
  f: Fields,
  mode: '2d' | '3d',
  variant: 'naive' | 'corrected',
  showUQ = false,
  opts: { height?: number; compact?: boolean } = {},
) {
  const s = exp.fieldSpec
  const zmin = minOf(f.ref)
  const zmax = maxOf(f.ref)
  const pad = (zmax - zmin) * 0.02
  const errMax = maxOf(f.err.naive)
  const z = kind === 'ref' ? f.ref : kind === 'pred' ? (variant === 'naive' ? f.naive : f.corrected) : variant === 'naive' ? f.err.naive : f.err.corrected
  const scale = kind === 'ref' ? SOL_SCALE : kind === 'pred' ? PRED_SCALE : ERR_SCALE
  const zr = kind === 'err' ? [0, errMax] : [zmin - pad, zmax + pad]
  const cbTitle = kind === 'err' ? '|Δu|' : s.uLabel
  const compact = opts.compact

  if (mode === '2d') {
    return {
      data: [
        {
          type: 'heatmap',
          x: f.x,
          y: f.y,
          z,
          zmin: zr[0],
          zmax: zr[1],
          colorscale: scale,
          zsmooth: 'best',
          colorbar: { thickness: 10, len: 0.9, outlinewidth: 0, tickfont: { size: compact ? 8 : 10 }, title: compact ? undefined : { text: cbTitle, side: 'right', font: { size: 10 } }, exponentformat: 'e' },
          hovertemplate: `${s.xLabel}=%{x:.3f}<br>${s.yLabel}=%{y:.3f}<br>${kind === 'err' ? '|Δu|' : 'u'}=%{z:.4g}<extra></extra>`,
        },
      ],
      layout: {
        height: opts.height ?? 300,
        margin: { l: 44, r: 8, t: 8, b: 38 },
        xaxis: { title: { text: s.xLabel, standoff: 4 }, showgrid: false, zeroline: false },
        yaxis: { title: { text: s.yLabel, standoff: 4 }, showgrid: false, zeroline: false },
        paper_bgcolor: 'rgba(0,0,0,0)',
        font: { ...FONT, size: compact ? 9 : 11 },
      },
    }
  }

  const data: any[] = [
    {
      type: 'surface',
      x: f.x,
      y: f.y,
      z,
      cmin: zr[0],
      cmax: zr[1],
      colorscale: scale,
      showscale: false,
      contours: { z: { show: false } },
      lighting: { ambient: 0.85, diffuse: 0.5, specular: 0.15, roughness: 0.6 },
    },
  ]
  if (showUQ && kind === 'pred') {
    for (const sign of [1, -1]) {
      data.push({
        type: 'surface',
        x: f.x,
        y: f.y,
        z: z.map((r, j) => r.map((v, i) => v + sign * 1.96 * f.std[j][i])),
        showscale: false,
        opacity: 0.18,
        colorscale: [[0, '#EC4899'], [1, '#EC4899']],
        hoverinfo: 'skip',
      })
    }
  }
  return {
    data,
    layout: {
      height: opts.height ?? 300,
      margin: { l: 0, r: 0, t: 0, b: 0 },
      paper_bgcolor: 'rgba(0,0,0,0)',
      font: { ...FONT, size: compact ? 9 : 11 },
      scene: {
        xaxis: { title: s.xLabel, backgroundcolor: 'rgba(245,243,255,0.6)', gridcolor: '#DDD6FE', showbackground: true },
        yaxis: { title: s.yLabel, backgroundcolor: 'rgba(255,245,247,0.6)', gridcolor: '#FBCFE8', showbackground: true },
        zaxis: { title: kind === 'err' ? '|Δu|' : s.uLabel, backgroundcolor: 'rgba(236,254,255,0.6)', gridcolor: '#A5F3FC', showbackground: true, range: kind === 'err' ? [0, errMax] : undefined },
        camera: { eye: { x: 1.55, y: -1.55, z: 0.9 } },
        aspectmode: 'manual',
        aspectratio: { x: 1.25, y: 1, z: 0.65 },
      },
    },
  }
}

export function overlayFigure(exp: Experiment, f: Fields, slices: Slice[], showUQ: boolean, opts: { height?: number; compact?: boolean } = {}) {
  const s = exp.fieldSpec
  const data: any[] = []
  const annotations: any[] = []
  const layout: any = {
    height: opts.height ?? 420,
    margin: { l: 40, r: 10, t: 20, b: 36 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    font: { ...FONT, size: opts.compact ? 9 : 11 },
    legend: { orientation: 'h', y: -0.12, x: 0.5, xanchor: 'center', font: { size: opts.compact ? 9 : 11 } },
    annotations,
  }
  const gapX = 0.09
  const gapY = 0.16
  const w = (1 - gapX) / 2
  const h = (1 - gapY) / 2
  slices.forEach((sl, k) => {
    const col = k % 2
    const row = Math.floor(k / 2)
    const xa = k === 0 ? 'x' : `x${k + 1}`
    const ya = k === 0 ? 'y' : `y${k + 1}`
    const xk = k === 0 ? 'xaxis' : `xaxis${k + 1}`
    const yk = k === 0 ? 'yaxis' : `yaxis${k + 1}`
    const x0 = col * (w + gapX)
    const y1 = 1 - row * (h + gapY)
    layout[xk] = { domain: [x0, x0 + w], anchor: ya, showgrid: true, gridcolor: '#EDE9FE', zeroline: false, title: row === 1 ? { text: s.xLabel, standoff: 2 } : undefined }
    layout[yk] = { domain: [y1 - h, y1], anchor: xa, showgrid: true, gridcolor: '#EDE9FE', zeroline: false }
    annotations.push({ text: `<b>${sl.label}</b>`, x: x0 + w / 2, y: y1 + 0.045, xref: 'paper', yref: 'paper', showarrow: false, font: { size: 11, color: '#6D28D9' } })
    const uq = f.std[sl.iy].map((v) => 1.96 * v)
    const corr = f.corrected[sl.iy]
    const first = k === 0
    if (showUQ) {
      data.push({ x: f.x, y: corr.map((v, i) => v + uq[i]), xaxis: xa, yaxis: ya, mode: 'lines', line: { width: 0 }, hoverinfo: 'skip', showlegend: false })
      data.push({ x: f.x, y: corr.map((v, i) => v - uq[i]), xaxis: xa, yaxis: ya, mode: 'lines', line: { width: 0 }, fill: 'tonexty', fillcolor: 'rgba(236,72,153,0.22)', name: '95% UQ band', showlegend: first, legendgroup: 'uq', hoverinfo: 'skip' })
    }
    const line = (name: string, y: number[], color: string, dash: string, width: number, key: string) =>
      data.push({ x: f.x, y, xaxis: xa, yaxis: ya, mode: 'lines', name, legendgroup: key, showlegend: first, line: { color, dash, width }, hovertemplate: `${name}<br>${s.xLabel}=%{x:.3f}<br>u=%{y:.4g}<extra></extra>` })
    line('Numerical Reference', f.ref[sl.iy], METHOD_COLORS.ref, 'solid', 2.2, 'ref')
    line('Baseline PINN', f.naive[sl.iy], METHOD_COLORS.baseline, 'dot', 1.8, 'base')
    line('Agent-Corrected PINN', corr, METHOD_COLORS.agent, 'solid', 2, 'agent')
    line('1D FNO', f.fno[sl.iy], METHOD_COLORS.fno, 'dash', 1.6, 'fno')
  })
  return { data, layout }
}
