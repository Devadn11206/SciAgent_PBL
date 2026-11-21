import type { Experiment } from './types'

export interface Fields {
  x: number[]
  y: number[]
  ref: number[][]
  naive: number[][]
  corrected: number[][]
  fno: number[][]
  std: number[][] // 1-sigma ensemble std of the corrected model
  err: { naive: number[][]; corrected: number[][] }
}

export interface Slice {
  label: string
  iy: number
  yVal: number
}

const linspace = (a: number, b: number, n: number) => Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1))
const sech = (v: number) => 1 / Math.cosh(v)

function reference(kind: Experiment['fieldSpec']['kind'], x: number, y: number): number {
  switch (kind) {
    case 'burgers': {
      const t = y
      return (-Math.tanh(x * (1.2 + 14 * (1 - Math.exp(-3 * t)))) * 1) / (1 + 1.1 * t)
    }
    case 'allenCahn': {
      const a = 0.62 - 0.32 * y
      const w = 0.07 + 0.03 * y
      return Math.tanh((x + a) / w) - Math.tanh((x - a) / w) - 1
    }
    case 'nlse': {
      const z = y
      return 1.5 * sech(1.5 * (x - 0.8 * z)) * (1 + 0.05 * Math.cos(6 * z))
    }
    case 'duffing': {
      const t = x
      const g = y
      const a = g * 2.6
      return (
        a * Math.cos(1.2 * t - 0.6) +
        0.45 * Math.exp(-0.08 * t) * Math.cos(0.95 * t + 3 * g) +
        0.12 * Math.sin(0.37 * t + 4 * g)
      )
    }
  }
}

function rms(f: number[][]) {
  let s = 0
  let n = 0
  for (const r of f) for (const v of r) { s += v * v; n++ }
  return Math.sqrt(s / n)
}

export function buildFields(exp: Experiment): Fields {
  const s = exp.fieldSpec
  const x = linspace(s.xRange[0], s.xRange[1], s.nx)
  const y = linspace(s.yRange[0], s.yRange[1], s.ny)
  const ref = y.map((yy) => x.map((xx) => reference(s.kind, xx, yy)))
  const maxAbs = Math.max(...ref.flat().map(Math.abs))
  const refRms = rms(ref)

  // gradient magnitude (normalised) — errors concentrate where the solution is sharp
  const grad = ref.map((row, j) =>
    row.map((_, i) => {
      const dx = (row[Math.min(i + 1, s.nx - 1)] - row[Math.max(i - 1, 0)]) / (x[Math.min(i + 1, s.nx - 1)] - x[Math.max(i - 1, 0)])
      const dy = (ref[Math.min(j + 1, s.ny - 1)][i] - ref[Math.max(j - 1, 0)][i]) / (y[Math.min(j + 1, s.ny - 1)] - y[Math.max(j - 1, 0)])
      return Math.hypot(dx, dy * 0.15)
    }),
  )
  const gmax = Math.max(...grad.flat())
  const gn = grad.map((r) => r.map((v) => v / gmax))

  const makeErr = (target: number, phase: number, freq: number, sharpWeight: number, growth: number) => {
    const raw = y.map((_, j) =>
      x.map((_, i) => {
        const xn = i / (s.nx - 1)
        const yn = j / (s.ny - 1)
        const wave = Math.sin(freq * 5.3 * xn + phase) * Math.cos(freq * 4.1 * yn + 0.7 * phase) + 0.35 * Math.sin(freq * 11 * xn - 3 * yn)
        const envelope = 0.25 + sharpWeight * gn[j][i]
        const grow = 1 + growth * (s.kind === 'duffing' ? xn : yn) * 2
        return wave * envelope * grow
      }),
    )
    const k = (target * refRms) / rms(raw)
    return raw.map((r) => r.map((v) => v * k))
  }

  const rel = (kind: 'baseline' | 'agent' | 'fno') => exp.bench.find((b) => b.kind === kind)!.relL2 as number
  const eNaive = makeErr(rel('baseline'), 0.4, 1.0, 1.4, s.errGrowth)
  const eCorr = makeErr(rel('agent'), 1.9, 1.0, 0.9, s.errGrowth * 0.5)
  const eFno = makeErr(rel('fno'), 3.1, 1.7, 0.6, s.errGrowth * 0.8)

  const add = (a: number[][], e: number[][]) => a.map((r, j) => r.map((v, i) => v + e[j][i]))
  const std = gn.map((r) => r.map((v) => (s.uqRel * maxAbs * (0.35 + 1.1 * v)) / 1.96))

  return {
    x,
    y,
    ref,
    naive: add(ref, eNaive),
    corrected: add(ref, eCorr),
    fno: add(ref, eFno),
    std,
    err: { naive: eNaive.map((r) => r.map(Math.abs)), corrected: eCorr.map((r) => r.map(Math.abs)) },
  }
}

export function buildSlices(exp: Experiment, f: Fields): Slice[] {
  const s = exp.fieldSpec
  return s.sliceFractions.map((fr) => {
    const target = s.yRange[0] + fr * (s.yRange[1] - s.yRange[0])
    let iy = 0
    let best = Infinity
    f.y.forEach((v, i) => {
      const d = Math.abs(v - target)
      if (d < best) { best = d; iy = i }
    })
    const yVal = f.y[iy]
    const fmt = s.sliceVar === 'γ' ? yVal.toFixed(2) : yVal.toFixed(2)
    return { label: `${s.sliceVar} = ${fmt}`, iy, yVal }
  })
}

/* ------------------------------ loss curves ------------------------------- */

export interface LossPoint {
  epoch: number
  pde: number
  bc: number
  data: number
  val: number
}

function mulberry(seed: number) {
  let a = seed
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export function buildLosses(exp: Experiment): LossPoint[] {
  const ls = exp.lossSpec
  const rand = mulberry(exp.id.length * 977 + Math.round(ls.initial * 100))
  const pts: LossPoint[] = []
  let v = Math.log10(ls.initial)
  let phase = 0
  for (let e = 0; e <= ls.epochs; e += ls.step) {
    while (phase < ls.interventions.length && e >= ls.interventions[phase].epoch) {
      phase++
      v += 0.45 // loss bump when the Critic changes the problem (new points / new weights)
    }
    const target = Math.log10(ls.plateaus[phase])
    v += (target - v) * 0.055 + (rand() - 0.5) * 0.045
    const noise = () => (rand() - 0.5) * 0.06
    const pde = Math.pow(10, v)
    pts.push({
      epoch: e,
      pde,
      bc: Math.pow(10, v - 0.45 + noise()),
      data: Math.pow(10, v - 0.2 + noise()),
      val: Math.pow(10, v + (phase === 0 ? 0.2 : 0.1) + noise()),
    })
  }
  return pts
}
