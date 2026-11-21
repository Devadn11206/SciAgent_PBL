export interface ExperimentMeta {
  id: string
  name: string
  title: string
  subtitle: string
  tint: string
  accent: string
  icon: 'waves' | 'battery' | 'fiber' | 'wing'
  mode: 'Forward' | 'Inverse'
  latex: string
  referenceSolver: string
}

export const PRESET_METAS: Record<string, ExperimentMeta> = {
  burgers: {
    id: 'burgers',
    name: "Viscous Burgers' Equation",
    title: "Viscous Burgers' Equation",
    subtitle: 'Shock wave formation · kinematic viscosity discovery',
    tint: 'from-fuchsia-50/70 to-pink-50/40',
    accent: '#DB2777',
    icon: 'waves',
    mode: 'Inverse',
    latex: '\\frac{\\partial u}{\\partial t} + u\\,\\frac{\\partial u}{\\partial x} = \\nu\\,\\frac{\\partial^{2} u}{\\partial x^{2}}',
    referenceSolver: 'Crank-Nicolson Finite Difference',
  },
  'allen-cahn': {
    id: 'allen-cahn',
    name: 'Allen-Cahn Equation',
    title: 'Allen-Cahn Phase Separation',
    subtitle: 'Interface transition · stiff cubic nonlinearity',
    tint: 'from-amber-50/70 to-orange-50/40',
    accent: '#D97706',
    icon: 'battery',
    mode: 'Forward',
    latex: '\\frac{\\partial u}{\\partial t} = \\varepsilon^{2}\\,\\frac{\\partial^{2} u}{\\partial x^{2}} - \\left(u^{3} - u\\right)',
    referenceSolver: 'Adaptive Crank-Nicolson FD',
  },
  nlse: {
    id: 'nlse',
    name: 'Cubic-Quintic NLSE',
    title: 'Cubic-Quintic NLSE',
    subtitle: 'Bright soliton propagation · optical nonlinear medium',
    tint: 'from-cyan-50/70 to-blue-50/40',
    accent: '#0891B2',
    icon: 'fiber',
    mode: 'Forward',
    latex: 'i\\,\\frac{\\partial \\psi}{\\partial t} + \\frac{\\beta}{2}\\,\\frac{\\partial^{2} \\psi}{\\partial x^{2}} + \\gamma\\,|\\psi|^{2}\\psi - \\delta\\,|\\psi|^{4}\\psi = 0',
    referenceSolver: 'Split-Step Fourier Method (SSFM)',
  },
  duffing: {
    id: 'duffing',
    name: 'Forced Duffing Oscillator',
    title: 'Forced Duffing Oscillator',
    subtitle: 'Nonlinear resonance & chaos · phase portraits',
    tint: 'from-violet-50/70 to-purple-50/40',
    accent: '#7C3AED',
    icon: 'wing',
    mode: 'Inverse',
    latex: '\\ddot{x} + \\delta\\,\\dot{x} + \\alpha\\,x + \\beta\\,x^{3} = \\gamma\\cos(\\omega t)',
    referenceSolver: 'Adaptive DOP853 Runge-Kutta',
  },
}

export interface PresetParamCatalogItem {
  symbol: string
  label: string
  token: string
  description: string
  min: number
  max: number
  step: number
  guess: number
  lo: number
  hi: number
}

export const PARAM_CATALOG: Record<string, PresetParamCatalogItem> = {
  nu: { symbol: 'nu', label: 'ν', token: '\\nu', description: 'Kinematic viscosity', min: 0.001, max: 0.1, step: 0.001, guess: 0.02, lo: 0.001, hi: 0.1 },
  eps: { symbol: 'eps', label: 'ε', token: '\\varepsilon', description: 'Interface thickness parameter', min: 0.01, max: 0.2, step: 0.005, guess: 0.05, lo: 0.01, hi: 0.2 },
  beta: { symbol: 'beta', label: 'β', token: '\\beta', description: 'Group velocity dispersion (GVD)', min: 0.1, max: 3.0, step: 0.1, guess: 1.0, lo: 0.1, hi: 3.0 },
  gamma: { symbol: 'gamma', label: 'γ', token: '\\gamma', description: 'Self-phase modulation (SPM) / driving amplitude', min: 0.1, max: 2.0, step: 0.05, guess: 0.5, lo: 0.1, hi: 2.0 },
  delta: { symbol: 'delta', label: 'δ', token: '\\delta', description: 'Quintic saturation / damping coefficient', min: 0.01, max: 1.0, step: 0.02, guess: 0.3, lo: 0.01, hi: 1.0 },
  alpha: { symbol: 'alpha', label: 'α', token: '\\alpha', description: 'Linear stiffness coefficient', min: -2.0, max: 2.0, step: 0.1, guess: -1.0, lo: -2.0, hi: 2.0 },
  omega: { symbol: 'omega', label: 'ω', token: '\\omega', description: 'External driving frequency', min: 0.2, max: 3.0, step: 0.1, guess: 1.2, lo: 0.2, hi: 3.0 },
}
