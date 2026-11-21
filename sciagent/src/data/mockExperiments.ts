import type {
  AblationRow,
  BenchRow,
  ConditionDef,
  Experiment,
  ExperimentId,
  HistoryEntry,
  LogLevel,
  LogLine,
  ParamDef,
} from './types'
import { sci } from './format'

/* -------------------------------------------------------------------------- */
/*  Helpers                                                                    */
/* -------------------------------------------------------------------------- */

const L = (stage: number, progress: number, level: LogLevel, text: string): LogLine => ({
  stage,
  progress,
  level,
  text,
})

const cond = (
  id: string,
  role: ConditionDef['role'],
  kind: ConditionDef['kind'],
  where: string,
  expr: string,
): ConditionDef => ({ id, role, kind, where, expr })

/** Orchestrator stage table (shared by all experiments)
 *  0 Problem Analysis · 1 Method Selection · 2 Execution Trial 1 · 3 Critic
 *  4 Execution Trial 2/3 (self-correct) · 5 Critic · 6 Execution Trial 3/3 · 7 Critic verify
 *  8 Report · 9 Done
 */

/** Catalogue used when the user types a new symbol into the LaTeX editor. */
export const PARAM_CATALOG: Record<string, Omit<ParamDef, 'mode' | 'guess' | 'lo' | 'hi' | 'ensemble' | 'sensorFile'> & { guess: number; lo: number; hi: number }> = {
  nu: { symbol: 'nu', label: 'ν', token: '\\nu', description: 'Kinematic viscosity', value: 0.01, min: 0.001, max: 0.1, step: 0.001, guess: 0.02, lo: 0.001, hi: 0.1 },
  eps: { symbol: 'eps', label: 'ε', token: '\\varepsilon', description: 'Interface-width parameter', value: 0.05, min: 0.01, max: 0.2, step: 0.005, guess: 0.08, lo: 0.01, hi: 0.2 },
  gamma: { symbol: 'gamma', label: 'γ', token: '\\gamma', description: 'Nonlinear coefficient', value: 1, min: 0, max: 3, step: 0.01, guess: 0.6, lo: 0.1, hi: 3 },
  delta: { symbol: 'delta', label: 'δ', token: '\\delta', description: 'Higher-order / damping coefficient', value: 0.15, min: 0, max: 1, step: 0.01, guess: 0.05, lo: 0, hi: 0.5 },
  beta: { symbol: 'beta', label: 'β', token: '\\beta', description: 'Cubic / dispersion coefficient', value: 1, min: 0.1, max: 3, step: 0.01, guess: 1, lo: 0.1, hi: 3 },
  alpha: { symbol: 'alpha', label: 'α', token: '\\alpha', description: 'Linear stiffness', value: -1, min: -2, max: 2, step: 0.05, guess: -0.5, lo: -2, hi: 2 },
  omega: { symbol: 'omega', label: 'ω', token: '\\omega', description: 'Driving frequency', value: 1.2, min: 0.5, max: 2, step: 0.05, guess: 1, lo: 0.5, hi: 2 },
}

/* -------------------------------------------------------------------------- */
/*  1 · Viscous Burgers — Traffic & Pipeline Shockwaves                       */
/* -------------------------------------------------------------------------- */

const burgersBench: BenchRow[] = [
  { method: 'Numerical Reference', detail: 'Crank–Nicolson, N=2048, Δt=1e-4', kind: 'reference', relL2: null, linf: null, meanRes: 4.1e-8, driftPct: 0.002, wallS: 3.8, memMB: 212 },
  { method: 'Baseline PINN (Trial 1)', detail: 'tanh MLP 6×64, Adam 4.2k ep', kind: 'baseline', relL2: 4.87e-2, linf: 3.41e-1, meanRes: 8.6e-3, driftPct: 2.9, wallS: 9.1, memMB: 640 },
  { method: 'Agent-Corrected PINN', detail: 'RAR + Causal + L-BFGS', kind: 'agent', relL2: 2.31e-3, linf: 1.87e-2, meanRes: 4.2e-5, driftPct: 0.14, wallS: 24.5, memMB: 702 },
  { method: '1D FNO', detail: '4 Fourier layers, 16 modes', kind: 'fno', relL2: 8.9e-3, linf: 6.2e-2, meanRes: 1.3e-3, driftPct: 0.61, wallS: 6.4, memMB: 488 },
]
const burgersAblation: AblationRow[] = [
  { trial: 'T1', action: 'Vanilla PINN (baseline)', diagnosis: 'Shock smearing at x≈0; 71% of residual mass in |x|<0.1', relL2: 4.87e-2 },
  { trial: 'T2', action: '+ Residual-Adaptive Refinement (RAR, +2000 pts)', diagnosis: 'Collocation density too low in shock layer', relL2: 1.62e-2 },
  { trial: 'T2', action: '+ Causal temporal weighting (ε=1e-2)', diagnosis: 'Late-time residuals dominate early-time errors', relL2: 6.9e-3 },
  { trial: 'T3', action: '+ L-BFGS fine-tune (2000 it, strong-Wolfe)', diagnosis: 'Adam stalls near 7e-3; curvature-aware polish', relL2: 2.31e-3 },
]

/* -------------------------------------------------------------------------- */
/*  2 · Allen–Cahn — Li-Ion Battery Phase Separation                          */
/* -------------------------------------------------------------------------- */

const acBench: BenchRow[] = [
  { method: 'Numerical Reference', detail: 'Strang split-step FFT, N=1024', kind: 'reference', relL2: null, linf: null, meanRes: 6.0e-9, driftPct: 0.001, wallS: 2.9, memMB: 187 },
  { method: 'Baseline PINN (Trial 1)', detail: 'tanh MLP 6×64, Adam 4.8k ep', kind: 'baseline', relL2: 1.12e-1, linf: 8.9e-1, meanRes: 2.4e-2, driftPct: 7.8, wallS: 11.4, memMB: 655 },
  { method: 'Agent-Corrected PINN', detail: 'Periodic features + Causal + RAR + L-BFGS', kind: 'agent', relL2: 3.84e-3, linf: 3.07e-2, meanRes: 9.7e-5, driftPct: 0.09, wallS: 31.2, memMB: 731 },
  { method: '1D FNO', detail: '4 Fourier layers, 20 modes', kind: 'fno', relL2: 1.74e-2, linf: 1.2e-1, meanRes: 2.1e-3, driftPct: 0.83, wallS: 7.9, memMB: 502 },
]
const acAblation: AblationRow[] = [
  { trial: 'T1', action: 'Vanilla PINN (baseline)', diagnosis: 'Collapse to trivial u≈−1 branch; interface lost after t>0.3', relL2: 1.12e-1 },
  { trial: 'T2', action: '+ Hard periodic embedding (Fourier features)', diagnosis: 'Boundary mismatch at x=±1 (|Δu|=0.41)', relL2: 4.31e-2 },
  { trial: 'T2', action: '+ Causal temporal weighting (ε=1e-3)', diagnosis: 'Optimizer fits late times before initial layer', relL2: 1.37e-2 },
  { trial: 'T3', action: '+ Residual-Adaptive Refinement (+2500 pts)', diagnosis: 'Residual spikes on two moving interfaces', relL2: 7.9e-3 },
  { trial: 'T3', action: '+ L-BFGS fine-tune (2000 it)', diagnosis: 'Residual floor set by first-order optimizer', relL2: 3.84e-3 },
]

/* -------------------------------------------------------------------------- */
/*  3 · Cubic-Quintic NLSE — Fiber-Optic Soliton Recovery (Inverse)           */
/* -------------------------------------------------------------------------- */

const nlseBench: BenchRow[] = [
  { method: 'Numerical Reference', detail: 'Split-step FFT, N=1024, Δt=1e-3', kind: 'reference', relL2: null, linf: null, meanRes: 7.0e-10, driftPct: 0.0003, wallS: 1.9, memMB: 164 },
  { method: 'Baseline PINN (Trial 1)', detail: 'Re/Im MLP 6×96, Adam 5k ep', kind: 'baseline', relL2: 6.35e-2, linf: 4.8e-1, meanRes: 1.4e-2, driftPct: 5.7, wallS: 21.3, memMB: 910 },
  { method: 'Agent-Corrected PINN', detail: 'Norm penalty + NTK balance + RAR + L-BFGS', kind: 'agent', relL2: 1.76e-3, linf: 1.54e-2, meanRes: 3.1e-5, driftPct: 0.08, wallS: 58.7, memMB: 986 },
  { method: '1D FNO', detail: '4 Fourier layers, 24 modes', kind: 'fno', relL2: 7.4e-3, linf: 5.5e-2, meanRes: 8.2e-4, driftPct: 0.47, wallS: 12.5, memMB: 640 },
]
const nlseAblation: AblationRow[] = [
  { trial: 'T1', action: 'Vanilla PINN (baseline)', diagnosis: 'Phase drift; amplitude decays 8% by t=1 (L2 norm violated)', relL2: 6.35e-2 },
  { trial: 'T2', action: '+ Re/Im split head + L2-norm conservation penalty', diagnosis: '∫|ψ|²dx drift 5.7% > 0.5% tolerance', relL2: 2.48e-2 },
  { trial: 'T2', action: '+ NTK / grad-norm loss balancing', diagnosis: 'Residual gradients 40× larger than IC gradients', relL2: 9.1e-3 },
  { trial: 'T3', action: '+ Residual-Adaptive Refinement (+2000 pts)', diagnosis: 'Residual concentrates under the soliton core', relL2: 4.2e-3 },
  { trial: 'T3', action: '+ L-BFGS fine-tune (2000 it)', diagnosis: 'γ, δ coupled — polish to resolve degeneracy', relL2: 1.76e-3 },
]

/* -------------------------------------------------------------------------- */
/*  4 · Forced Duffing — Wing Flutter & Grid Chaos                            */
/* -------------------------------------------------------------------------- */

const duffBench: BenchRow[] = [
  { method: 'Numerical Reference', detail: 'DOP853, rtol=1e-12', kind: 'reference', relL2: null, linf: null, meanRes: 1.0e-12, driftPct: 0.0000001, wallS: 0.4, memMB: 96 },
  { method: 'Baseline PINN (Trial 1)', detail: 'tanh MLP 5×64, Adam 6k ep', kind: 'baseline', relL2: 2.84e-1, linf: 1.9, meanRes: 3.7e-2, driftPct: 12.4, wallS: 14.8, memMB: 520 },
  { method: 'Agent-Corrected PINN', detail: 'Time-marching + Fourier + Ensemble + L-BFGS', kind: 'agent', relL2: 5.62e-3, linf: 4.1e-2, meanRes: 8.8e-5, driftPct: 0.21, wallS: 41.8, memMB: 585 },
  { method: '1D FNO', detail: '4 Fourier layers, 12 modes', kind: 'fno', relL2: 4.07e-2, linf: 2.6e-1, meanRes: 4.4e-3, driftPct: 1.9, wallS: 9.8, memMB: 410 },
]
const duffAblation: AblationRow[] = [
  { trial: 'T1', action: 'Vanilla PINN (baseline)', diagnosis: 'Phase lag grows after t≈15; converges to trivial damped orbit', relL2: 2.84e-1 },
  { trial: 'T2', action: '+ Time-marching windows (4 × Δt=10)', diagnosis: 'Long-horizon forced oscillation not trainable in one window', relL2: 6.1e-2 },
  { trial: 'T2', action: '+ ω-locked Fourier time features', diagnosis: 'Spectral bias: forcing harmonic at ω=1.2 under-resolved', relL2: 2.17e-2 },
  { trial: 'T3', action: '+ Ensemble ×8 with β bound clamp [0.5, 3]', diagnosis: 'β posterior bimodal in Trial 2 (1.1 and 1.5)', relL2: 1.18e-2 },
  { trial: 'T3', action: '+ L-BFGS fine-tune (2000 it)', diagnosis: 'Residual floor set by first-order optimizer', relL2: 5.62e-3 },
]

/* -------------------------------------------------------------------------- */
/*  Experiments                                                                */
/* -------------------------------------------------------------------------- */

const b = (rows: BenchRow[], kind: BenchRow['kind']) => rows.find((r) => r.kind === kind)!

export const experiments: Experiment[] = [
  /* ------------------------------ BURGERS --------------------------------- */
  {
    id: 'burgers',
    title: 'Traffic & Pipeline Shockwaves',
    subtitle: "Viscous Burgers' PDE",
    tint: 'from-cyan-50 via-white/80 to-white/70',
    accent: '#06B6D4',
    icon: 'waves',
    pdeName: "Viscous Burgers' Equation",
    mode: 'Inverse',
    isODE: false,
    latex: String.raw`\frac{\partial u}{\partial t} + u\,\frac{\partial u}{\partial x} = \nu\,\frac{\partial^{2} u}{\partial x^{2}}`,
    referenceSolver: 'Crank–Nicolson',
    domainLabels: { x: 'x (position)', t: 't (time)' },
    domain: { xMin: -1, xMax: 1, tMin: 0, tMax: 1 },
    uScale: 1.0,
    conditions: [
      cond('ic', 'Initial', 'Dirichlet', 't = 0', 'u(x, 0) = −sin(πx)'),
      cond('bc1', 'Boundary', 'Dirichlet', 'x = −1', 'u(−1, t) = 0'),
      cond('bc2', 'Boundary', 'Dirichlet', 'x = +1', 'u(+1, t) = 0'),
    ],
    invariant: 'L2 Energy Conservation',
    params: [
      { ...PARAM_CATALOG.nu, mode: 'inverse', ensemble: true, sensorFile: 'loop_detectors_n64.csv', truth: 0.01, value: 0.01 } as ParamDef,
    ],
    kpis: {
      relL2: '2.31e-3',
      driftLabel: 'Energy Drift',
      drift: '0.14%',
      paramLabel: 'Discovered ν',
      param: '0.01003 ± 0.00021',
      runtime: '24.5s',
    },
    logs: [
      L(0, 0, 'AGENT', 'Problem Analysis Agent · parsing LaTeX → SymPy AST …'),
      L(0, 0, 'SYMPY', 'u_t + u·u_x − ν·u_xx = 0 | order 2 | quasi-linear | 1 unknown coeff (ν)'),
      L(0, 0, 'SYMPY', 'CFL pre-flight: max|u|=1.00, Δx=1/32 → Δt ≤ 3.1e-2; diffusive Δt ≤ Δx²/(2ν)=5.1e-2 ✓'),
      L(0, 0, 'WARN', 'Shock-formation estimate t* = 1/max|u_x(0)| = 0.318 < t_max → flag adaptive sampling'),
      L(1, 0, 'AGENT', 'Method Selection Agent · candidates {PINN, FNO, Crank–Nicolson}'),
      L(1, 0, 'INFO', 'selected: CN reference (N=2048) + PINN tanh 6×64 + 1D FNO baseline · ensemble=8 for ν'),
      L(1, 0, 'OK', 'Conservation audit armed: d/dt ∫u²/2 dx = −ν∫u_x² dx tracked on 101 time-slices'),
      L(2, 0.2, 'INFO', 'Execution Engine · Trial 1/3 · Adam lr=1e-3 · N_f=10k collocation pts'),
      L(2, 0.35, 'WARN', 'Trial 1 stagnation: residual loss plateau 8.6e-3, validation 1.1e-2'),
      L(3, 0.4, 'CRITIC', 'Rel L2 = 4.87e-2 > tol 1e-2 · 71% of residual mass inside |x|<0.1 (shock layer)'),
      L(3, 0.4, 'CRITIC', 'Conservation audit: energy-balance drift 2.9% > 0.5% tol → FAIL'),
      L(3, 0.4, 'CRITIC', 'Action plan: [RAR +2000 pts] [Causal weighting ε=1e-2] → re-run (Trial 2/3)'),
      L(4, 0.55, 'INFO', 'Self-Correcting Trial 2/3 · RAR injected 2000 pts, 83% inside |x|<0.12'),
      L(4, 0.7, 'OK', 'Trial 2/3 · Rel L2 4.87e-2 → 6.9e-3 · energy drift 0.62%'),
      L(5, 0.75, 'CRITIC', 'Still above target 5e-3 → schedule L-BFGS fine-tune (2000 it, strong-Wolfe)'),
      L(6, 0.88, 'INFO', 'Self-Correcting Trial 3/3 · L-BFGS …'),
      L(6, 0.95, 'OK', 'Trial 3/3 · Rel L2 2.31e-3 ✓ · ν = 0.01003 ± 0.00021 (ensemble n=8)'),
      L(7, 0.98, 'CRITIC', 'Verification: energy drift 0.14% ✓ · mean PDE residual 4.2e-5 ✓ → ACCEPT'),
      L(8, 1, 'AGENT', 'Research Report Agent · compiling single-column manuscript …'),
      L(9, 1, 'OK', 'Verified convergence · 3 trials · wall-clock 24.5s'),
    ],
    insight:
      'The Critic traced the naive failure to under-sampled collocation points inside the viscous shock layer. Residual-adaptive refinement plus causal weighting cut error 7×, and L-BFGS closed the remaining gap, recovering ν within 0.3% of ground truth.',
    agentNotes: [
      'Parsed the PDE into a SymPy AST, classified it as quasi-linear order-2, and estimated shock formation at t*≈0.318.',
      'Chose Crank–Nicolson as the numerical truth, a tanh PINN as the learner, and a 1D FNO as a data-driven baseline.',
      'Runs the chosen solver and trains the PINN trial by trial; re-entered by the Critic via the rose feedback loop.',
      'Audits error, residual spatial distribution and the energy-dissipation balance; prescribes RAR / Causal / L-BFGS actions.',
      'Writes the manuscript from verified metrics only — every sentence is tied to a measured number.',
    ],
    fieldSpec: {
      kind: 'burgers', nx: 61, ny: 41, xRange: [-1, 1], yRange: [0, 1],
      xLabel: 'x', yLabel: 't', uLabel: 'u(x,t)', sliceVar: 't',
      sliceFractions: [0.25, 0.5, 0.75, 1.0], errGrowth: 0, uqRel: 0.014,
    },
    lossSpec: {
      epochs: 12000, step: 200, initial: 2.0,
      interventions: [
        { epoch: 4200, label: 'Critic: RAR + Causal' },
        { epoch: 9000, label: 'Critic: L-BFGS' },
      ],
      plateaus: [8.6e-3, 6.0e-4, 4.2e-5],
    },
    bench: burgersBench,
    ablation: burgersAblation,
    report: {
      title: "Agent-Driven Physics-Informed Recovery of Viscous Shock Dynamics in Burgers' Flow",
      subtitle: 'Traffic & Pipeline Shockwave Case Study · Inverse Viscosity Identification',
      keywords: ['Burgers equation', 'PINN', 'shock capturing', 'inverse problem', 'residual-adaptive refinement'],
      findings: [
        `The agent-corrected PINN reaches a relative L2 error of ${sci(b(burgersBench, 'agent').relL2)}, a ${(b(burgersBench, 'baseline').relL2! / b(burgersBench, 'agent').relL2!).toFixed(1)}× reduction over the naive Trial 1 baseline (${sci(b(burgersBench, 'baseline').relL2)}).`,
        'The viscosity was recovered as ν = 0.01003 ± 0.00021 against a ground truth of 0.0100 (0.3% error, ensemble n = 8).',
        `The energy-dissipation balance holds to ${b(burgersBench, 'agent').driftPct}% drift, versus ${b(burgersBench, 'baseline').driftPct}% for the baseline.`,
      ],
      interpretation: [
        `Physically, the flow steepens into a viscous shock near x = 0 at t* ≈ 0.32, where convective steepening is balanced by diffusion. The naive PINN smeared this layer, and ${'71%'} of its residual mass sat within |x| < 0.1 — exactly where the Critic Agent concentrated additional collocation points.`,
        `Because the exact Burgers dynamics dissipate energy at the rate −ν∫u_x² dx, an energy-balance audit is a sharp physical test: the corrected model keeps this balance within ${b(burgersBench, 'agent').driftPct}%, indicating the learned shock is physically admissible rather than a numerical artifact.`,
        `The 1D FNO is ${(b(burgersBench, 'fno').relL2! / b(burgersBench, 'agent').relL2!).toFixed(1)}× less accurate (${sci(b(burgersBench, 'fno').relL2)}) but ${(b(burgersBench, 'agent').wallS / b(burgersBench, 'fno').wallS).toFixed(1)}× cheaper to train, making it a useful surrogate once the PINN has verified the physics.`,
      ],
      methods:
        'Reference: Crank–Nicolson with Newton linearisation on a 2048-point grid. PINN: tanh MLP 6×64, Adam (lr 1e-3) followed by L-BFGS. Uncertainty: deep ensemble of 8 members; 95% bands are ±1.96σ.',
      eqNote: 'Viscous Burgers equation with unknown viscosity ν on x ∈ [−1, 1], t ∈ [0, 1].',
    },
  },

  /* ---------------------------- ALLEN–CAHN -------------------------------- */
  {
    id: 'allen-cahn',
    title: 'Li-Ion Battery Phase Separation',
    subtitle: 'Allen–Cahn PDE',
    tint: 'from-pink-50 via-white/80 to-white/70',
    accent: '#EC4899',
    icon: 'battery',
    pdeName: 'Allen–Cahn Equation',
    mode: 'Inverse',
    isODE: false,
    latex: String.raw`\frac{\partial u}{\partial t} = \varepsilon^{2}\,\frac{\partial^{2} u}{\partial x^{2}} - \left(u^{3} - u\right)`,
    referenceSolver: 'Split-Step FFT',
    domainLabels: { x: 'x (electrode depth)', t: 't (time)' },
    domain: { xMin: -1, xMax: 1, tMin: 0, tMax: 1 },
    uScale: 1.0,
    conditions: [
      cond('ic', 'Initial', 'Dirichlet', 't = 0', 'u(x, 0) = x²·cos(πx)'),
      cond('bc1', 'Boundary', 'Periodic', 'x = ±1', 'u(−1, t) = u(+1, t)'),
      cond('bc2', 'Boundary', 'Periodic', 'x = ±1', 'uₓ(−1, t) = uₓ(+1, t)'),
    ],
    invariant: 'None',
    params: [
      { ...PARAM_CATALOG.eps, mode: 'inverse', ensemble: true, sensorFile: 'xray_phase_front_n48.csv', truth: 0.05, value: 0.05 } as ParamDef,
    ],
    kpis: {
      relL2: '3.84e-3',
      driftLabel: 'Free-Energy Drift',
      drift: '0.09%',
      paramLabel: 'Discovered ε',
      param: '0.0502 ± 0.0009',
      runtime: '31.2s',
    },
    logs: [
      L(0, 0, 'AGENT', 'Problem Analysis Agent · parsing LaTeX → SymPy AST …'),
      L(0, 0, 'SYMPY', 'u_t − ε²·u_xx + u³ − u = 0 | order 2 | semilinear, cubic source | 1 unknown coeff (ε)'),
      L(0, 0, 'SYMPY', 'CFL pre-flight: stiff reaction rate |f′(u)|max = 2 → Δt ≤ 1.0; diffusion Δt ≤ Δx²/(2ε²) = 0.2 ✓ (IMEX)'),
      L(0, 0, 'WARN', 'Known PINN failure mode: trivial u≡−1 minimizer for t > 0.3 → enable causal training'),
      L(1, 0, 'AGENT', 'Method Selection Agent · candidates {PINN, FNO, split-step FFT}'),
      L(1, 0, 'INFO', 'selected: Strang split-step FFT (N=1024) reference + PINN tanh 6×64 + FNO baseline'),
      L(1, 0, 'OK', 'Free-energy audit armed: E[u]=∫(ε²u_x²/2 + (u²−1)²/4)dx must decrease monotonically'),
      L(2, 0.2, 'INFO', 'Execution Engine · Trial 1/3 · Adam lr=1e-3 · N_f=12k collocation pts'),
      L(2, 0.35, 'WARN', 'Trial 1 collapse: u→−1 branch, interface lost after t=0.31'),
      L(3, 0.4, 'CRITIC', 'Rel L2 = 1.12e-1 > tol 1e-2 · periodic mismatch |u(−1)−u(1)| = 0.41'),
      L(3, 0.4, 'CRITIC', 'Free-energy audit: 7.8% non-monotone drift → FAIL'),
      L(3, 0.4, 'CRITIC', 'Action plan: [Periodic Fourier embedding] [Causal ε=1e-3] [RAR] → Trial 2/3'),
      L(4, 0.55, 'INFO', 'Self-Correcting Trial 2/3 · hard periodic embedding enabled; causal weights annealed'),
      L(4, 0.7, 'OK', 'Trial 2/3 · Rel L2 1.12e-1 → 7.9e-3 after RAR (+2500 pts on both interfaces)'),
      L(5, 0.75, 'CRITIC', 'Residual floor 1.1e-4 limited by first-order optimizer → schedule L-BFGS'),
      L(6, 0.88, 'INFO', 'Self-Correcting Trial 3/3 · L-BFGS …'),
      L(6, 0.95, 'OK', 'Trial 3/3 · Rel L2 3.84e-3 ✓ · ε = 0.0502 ± 0.0009 (ensemble n=8)'),
      L(7, 0.98, 'CRITIC', 'Verification: free-energy drift 0.09% ✓ · mean PDE residual 9.7e-5 ✓ → ACCEPT'),
      L(8, 1, 'AGENT', 'Research Report Agent · compiling single-column manuscript …'),
      L(9, 1, 'OK', 'Verified convergence · 3 trials · wall-clock 31.2s'),
    ],
    insight:
      'Plain PINNs collapse to the trivial u ≈ −1 state for Allen–Cahn. The Critic identified a periodic boundary mismatch and causality violation, then restored two sharp, correctly moving phase interfaces and ε within 0.4% of truth.',
    agentNotes: [
      'Classified the source term as stiff cubic reaction; flagged the known trivial-minimizer failure mode ahead of training.',
      'Selected an FFT split-step reference because it respects periodic boundaries exactly; PINN and FNO as learners.',
      'Executes IMEX reference and PINN trials; re-entered up to three times by the Critic.',
      'Audits periodicity, causality and monotone free-energy decay; prescribes embedding + causal + RAR + L-BFGS.',
      'Reports interface coarsening dynamics, tying every claim to a verified metric.',
    ],
    fieldSpec: {
      kind: 'allenCahn', nx: 61, ny: 41, xRange: [-1, 1], yRange: [0, 1],
      xLabel: 'x', yLabel: 't', uLabel: 'u(x,t)', sliceVar: 't',
      sliceFractions: [0.25, 0.5, 0.75, 1.0], errGrowth: 0.3, uqRel: 0.02,
    },
    lossSpec: {
      epochs: 12000, step: 200, initial: 1.5,
      interventions: [
        { epoch: 4800, label: 'Critic: Periodic + Causal + RAR' },
        { epoch: 9200, label: 'Critic: L-BFGS' },
      ],
      plateaus: [2.4e-2, 1.8e-3, 9.7e-5],
    },
    bench: acBench,
    ablation: acAblation,
    report: {
      title: 'Self-Correcting Physics-Informed Learning of Allen–Cahn Phase Separation in Li-Ion Electrodes',
      subtitle: 'Battery Phase-Separation Case Study · Inverse Interface-Width Identification',
      keywords: ['Allen–Cahn', 'phase field', 'causal PINN', 'periodic boundary', 'battery electrodes'],
      findings: [
        `Relative L2 error falls from ${sci(b(acBench, 'baseline').relL2)} (naive) to ${sci(b(acBench, 'agent').relL2)} after agentic self-correction — a ${(b(acBench, 'baseline').relL2! / b(acBench, 'agent').relL2!).toFixed(0)}× improvement.`,
        'The interface-width parameter ε is identified as 0.0502 ± 0.0009 versus ground truth 0.0500.',
        `Free energy decays monotonically with ${b(acBench, 'agent').driftPct}% residual drift, versus ${b(acBench, 'baseline').driftPct}% in Trial 1.`,
      ],
      interpretation: [
        'Phase separation in Allen–Cahn proceeds by sharpening of diffuse interfaces of width ≈ √2·ε followed by slow curvature-driven coarsening. The naive PINN converged to the trivial u ≡ −1 branch because the late-time residual dominated optimisation before the initial interface structure was learned.',
        `Enforcing periodicity through Fourier embeddings removed a boundary mismatch of 0.41, and causal weighting restored the correct temporal order of learning. The resulting free-energy drift of ${b(acBench, 'agent').driftPct}% shows the learned dynamics respect the gradient-flow structure of the equation.`,
        `The 1D FNO reaches ${sci(b(acBench, 'fno').relL2)} but violates monotone free-energy decay by ${b(acBench, 'fno').driftPct}%, so it is best used as a fast surrogate only after PINN verification.`,
      ],
      methods:
        'Reference: Strang split-step FFT on 1024 points with exact linear propagator. PINN: periodic Fourier-feature tanh MLP 6×64, Adam then L-BFGS. UQ: deep ensemble of 8.',
      eqNote: 'Allen–Cahn equation with unknown interface parameter ε on x ∈ [−1, 1], t ∈ [0, 1] (periodic).',
    },
  },

  /* ------------------------------- NLSE ----------------------------------- */
  {
    id: 'nlse',
    title: 'Fiber-Optic Soliton Recovery',
    subtitle: 'Cubic-Quintic NLSE · Inverse Mode',
    tint: 'from-violet-50 via-white/80 to-white/70',
    accent: '#7C3AED',
    icon: 'fiber',
    pdeName: 'Cubic-Quintic Nonlinear Schrödinger Equation',
    mode: 'Inverse',
    isODE: false,
    latex: String.raw`i\,\frac{\partial \psi}{\partial t} + \frac{\beta}{2}\,\frac{\partial^{2} \psi}{\partial x^{2}} + \gamma\,|\psi|^{2}\psi - \delta\,|\psi|^{4}\psi = 0`,
    referenceSolver: 'Split-Step FFT',
    domainLabels: { x: 'x (retarded time)', t: 'z (propagation)' },
    domain: { xMin: -6, xMax: 6, tMin: 0, tMax: 1 },
    uScale: 1.5,
    conditions: [
      cond('ic', 'Initial', 'Dirichlet', 't = 0', 'ψ(x, 0) = 1.5·sech(1.5x)'),
      cond('bc1', 'Boundary', 'Periodic', 'x = ±6', 'ψ(−6, t) = ψ(+6, t)'),
      cond('bc2', 'Boundary', 'Neumann', 'x = ±6', 'ψₓ(±6, t) ≈ 0 (soliton tails)'),
    ],
    invariant: 'L2 Energy Conservation',
    params: [
      { ...PARAM_CATALOG.beta, mode: 'fixed', value: 1, ensemble: false, sensorFile: null } as ParamDef,
      { ...PARAM_CATALOG.gamma, mode: 'inverse', ensemble: true, sensorFile: 'fiber_intensity_probe_n96.csv', truth: 1.0, value: 1.0 } as ParamDef,
      { ...PARAM_CATALOG.delta, mode: 'inverse', ensemble: true, sensorFile: 'fiber_intensity_probe_n96.csv', truth: 0.15, value: 0.15 } as ParamDef,
    ],
    kpis: {
      relL2: '1.76e-3',
      driftLabel: 'L2-Norm Drift',
      drift: '0.08%',
      paramLabel: 'Discovered γ',
      param: '1.003 ± 0.009',
      runtime: '58.7s',
    },
    logs: [
      L(0, 0, 'AGENT', 'Problem Analysis Agent · parsing complex-valued LaTeX → SymPy AST …'),
      L(0, 0, 'SYMPY', 'iψ_t + (β/2)ψ_xx + γ|ψ|²ψ − δ|ψ|⁴ψ = 0 | complex field → (Re ψ, Im ψ) | 2 unknown coeffs (γ, δ)'),
      L(0, 0, 'SYMPY', 'Split-step pre-flight: unitary linear step; nonlinear phase γ|ψ|²max·Δt = 2.3e-3 rad ≪ 1 at Δt=1e-3 ✓'),
      L(0, 0, 'WARN', 'Parameter degeneracy: γ and δ jointly control soliton width → require amplitude + width sensors'),
      L(1, 0, 'AGENT', 'Method Selection Agent · candidates {PINN, FNO, split-step FFT}'),
      L(1, 0, 'INFO', 'selected: Split-step FFT reference + Re/Im PINN tanh 6×96 + FNO baseline · ensemble=8 for (γ, δ)'),
      L(1, 0, 'OK', 'Conservation audit armed: N[ψ] = ∫|ψ|² dx must remain constant (Hamiltonian flow)'),
      L(2, 0.2, 'INFO', 'Execution Engine · Trial 1/3 · Adam lr=1e-3 · N_f=15k · 96 intensity probes'),
      L(2, 0.35, 'WARN', 'Trial 1: soliton amplitude decays 8% by z=1; phase drift visible in Im ψ'),
      L(3, 0.4, 'CRITIC', 'Rel L2 = 6.35e-2 > tol 1e-2 · γ estimate 0.81, δ estimate 0.29 (degenerate)'),
      L(3, 0.4, 'CRITIC', 'Conservation audit: L2-norm drift 5.7% > 0.5% tol → FAIL'),
      L(3, 0.4, 'CRITIC', 'Action plan: [norm penalty] [NTK balance] [RAR] → Trial 2/3'),
      L(4, 0.55, 'INFO', 'Self-Correcting Trial 2/3 · conservation penalty λ=5.0 · NTK weights rebalanced (IC:PDE = 40:1)'),
      L(4, 0.7, 'OK', 'Trial 2/3 · Rel L2 6.35e-2 → 4.2e-3 · γ=0.994, δ=0.157'),
      L(5, 0.75, 'CRITIC', 'γ–δ coupling remains · schedule L-BFGS fine-tune to resolve degeneracy'),
      L(6, 0.88, 'INFO', 'Self-Correcting Trial 3/3 · L-BFGS …'),
      L(6, 0.95, 'OK', 'Trial 3/3 · Rel L2 1.76e-3 ✓ · γ = 1.003 ± 0.009 · δ = 0.149 ± 0.006'),
      L(7, 0.98, 'CRITIC', 'Verification: norm drift 0.08% ✓ · mean PDE residual 3.1e-5 ✓ → ACCEPT'),
      L(8, 1, 'AGENT', 'Research Report Agent · compiling single-column manuscript …'),
      L(9, 1, 'OK', 'Verified convergence · 3 trials · wall-clock 58.7s'),
    ],
    insight:
      'The naive PINN could not separate the cubic and quintic nonlinearities and let the pulse energy leak. Enforcing L2-norm conservation and rebalancing loss gradients identified both coefficients, γ = 1.003 and δ = 0.149, within 1% of the fiber ground truth.',
    agentNotes: [
      'Converted ψ into real/imaginary channels and flagged γ–δ degeneracy that requires both amplitude and width sensor data.',
      'Selected split-step FFT as the unitary reference, a Re/Im PINN for inverse learning and an FNO surrogate.',
      'Runs reference propagation and PINN training; the feedback loop re-enters it with penalties and refined points.',
      'Audits ∫|ψ|²dx conservation, gradient balance and parameter posterior width; prescribes fixes.',
      'Interprets the recovered nonlinearity coefficients in terms of soliton width and peak power.',
    ],
    fieldSpec: {
      kind: 'nlse', nx: 71, ny: 41, xRange: [-6, 6], yRange: [0, 1],
      xLabel: 'x', yLabel: 'z', uLabel: '|ψ(x,z)|', sliceVar: 'z',
      sliceFractions: [0.25, 0.5, 0.75, 1.0], errGrowth: 0.4, uqRel: 0.018,
    },
    lossSpec: {
      epochs: 12000, step: 200, initial: 3.0,
      interventions: [
        { epoch: 3800, label: 'Critic: Norm penalty + NTK' },
        { epoch: 8600, label: 'Critic: L-BFGS' },
      ],
      plateaus: [1.4e-2, 9.0e-4, 3.1e-5],
    },
    bench: nlseBench,
    ablation: nlseAblation,
    report: {
      title: 'Inverse Recovery of Cubic-Quintic Soliton Nonlinearities with a Self-Correcting PINN Agent',
      subtitle: 'Fiber-Optic Soliton Case Study · Joint Identification of (γ, δ)',
      keywords: ['nonlinear Schrödinger', 'optical soliton', 'inverse PINN', 'norm conservation', 'uncertainty quantification'],
      findings: [
        `Relative L2 error on |ψ| improves from ${sci(b(nlseBench, 'baseline').relL2)} to ${sci(b(nlseBench, 'agent').relL2)} (${(b(nlseBench, 'baseline').relL2! / b(nlseBench, 'agent').relL2!).toFixed(0)}× reduction).`,
        'Both nonlinear coefficients were recovered: γ = 1.003 ± 0.009 (truth 1.000) and δ = 0.149 ± 0.006 (truth 0.150).',
        `The conserved L2 norm drifts by only ${b(nlseBench, 'agent').driftPct}% across the propagation window.`,
      ],
      interpretation: [
        'Cubic-quintic solitons arise from balance between anomalous dispersion, Kerr self-focusing (γ) and quintic saturation (δ). Because γ and δ jointly set pulse width, an unconstrained inverse PINN is ill-posed; the Critic therefore demanded amplitude and width sensor data and a hard norm penalty.',
        `With the L2-norm penalty active, the corrected model conserves optical energy to ${b(nlseBench, 'agent').driftPct}%, whereas the baseline leaked ${b(nlseBench, 'baseline').driftPct}% of pulse energy — an unphysical dissipation for a Hamiltonian flow.`,
        `The posterior widths (±0.009 on γ, ±0.006 on δ) show the inverse problem is well-conditioned after correction. The FNO baseline achieves ${sci(b(nlseBench, 'fno').relL2)} but offers no coefficient identification.`,
      ],
      methods:
        'Reference: Strang split-step Fourier method, N = 1024, Δt = 1e-3. PINN: two-channel (Re ψ, Im ψ) tanh MLP 6×96. Inverse variables γ, δ learned with log-parameterisation; ensemble of 8.',
      eqNote: 'Cubic-quintic NLSE with unknown nonlinear coefficients (γ, δ), x ∈ [−6, 6], z ∈ [0, 1].',
    },
  },

  /* ------------------------------ DUFFING --------------------------------- */
  {
    id: 'duffing',
    title: 'Wing Flutter & Grid Chaos',
    subtitle: 'Forced Duffing ODE',
    tint: 'from-emerald-50 via-white/80 to-white/70',
    accent: '#10B981',
    icon: 'wing',
    pdeName: 'Forced Duffing Oscillator',
    mode: 'Inverse',
    isODE: true,
    latex: String.raw`\ddot{x} + \delta\,\dot{x} + \alpha\,x + \beta\,x^{3} = \gamma\cos(\omega t)`,
    referenceSolver: 'DOP853',
    domainLabels: { x: 'x (state bounds)', t: 't (time)' },
    domain: { xMin: -2, xMax: 2, tMin: 0, tMax: 40 },
    uScale: 1.2,
    conditions: [
      cond('ic1', 'Initial', 'Dirichlet', 't = 0', 'x(0) = 0.5'),
      cond('ic2', 'Initial', 'Neumann', 't = 0', 'ẋ(0) = 0'),
    ],
    invariant: 'None',
    params: [
      { ...PARAM_CATALOG.delta, mode: 'fixed', value: 0.3, min: 0, max: 1, step: 0.01, description: 'Linear damping', ensemble: false, sensorFile: null } as ParamDef,
      { ...PARAM_CATALOG.alpha, mode: 'fixed', value: -1, ensemble: false, sensorFile: null } as ParamDef,
      { ...PARAM_CATALOG.beta, mode: 'inverse', value: 1.5, min: 0.1, max: 3, step: 0.01, guess: 1.0, lo: 0.5, hi: 3, truth: 1.5, ensemble: true, sensorFile: 'strain_gauge_wing_n120.csv', description: 'Cubic stiffness' } as ParamDef,
      { ...PARAM_CATALOG.gamma, mode: 'fixed', value: 0.37, min: 0, max: 1, step: 0.01, description: 'Forcing amplitude', ensemble: false, sensorFile: null } as ParamDef,
      { ...PARAM_CATALOG.omega, mode: 'fixed', value: 1.2, ensemble: false, sensorFile: null } as ParamDef,
    ],
    kpis: {
      relL2: '5.62e-3',
      driftLabel: 'Energy-Balance Drift',
      drift: '0.21%',
      paramLabel: 'Discovered β',
      param: '1.498 ± 0.012',
      runtime: '41.8s',
    },
    logs: [
      L(0, 0, 'AGENT', 'Problem Analysis Agent · parsing LaTeX → SymPy AST …'),
      L(0, 0, 'SYMPY', 'ẍ + δẋ + αx + βx³ = γcos(ωt) | order 2 ODE | non-autonomous | 1 unknown coeff (β)'),
      L(0, 0, 'SYMPY', 'Stability pre-flight: Lyapunov estimate λ₁ ≈ +0.09 over t∈[0,40] → mild sensitivity; DOP853 rtol=1e-12 ✓'),
      L(0, 0, 'WARN', 'Long horizon (t=40, ≈7.6 forcing periods) exceeds single-window PINN trainability'),
      L(1, 0, 'AGENT', 'Method Selection Agent · candidates {PINN, FNO, DOP853}'),
      L(1, 0, 'INFO', 'selected: DOP853 reference + time-marching PINN (4 windows) + FNO baseline · ensemble=8 for β'),
      L(1, 0, 'OK', 'Energy-balance audit armed: dE/dt = −δẋ² + γcos(ωt)ẋ with E = ẋ²/2 + αx²/2 + βx⁴/4'),
      L(2, 0.2, 'INFO', 'Execution Engine · Trial 1/3 · Adam lr=1e-3 · N_f=8k · 120 strain-gauge samples'),
      L(2, 0.35, 'WARN', 'Trial 1: phase lag grows after t≈15; trajectory collapses to damped orbit'),
      L(3, 0.4, 'CRITIC', 'Rel L2 = 2.84e-1 > tol 1e-2 · β posterior bimodal (1.1 / 1.5)'),
      L(3, 0.4, 'CRITIC', 'Energy-balance audit: drift 12.4% > 1% tol → FAIL'),
      L(3, 0.4, 'CRITIC', 'Action plan: [time-marching windows] [ω-locked Fourier features] [Ensemble ×8] → Trial 2/3'),
      L(4, 0.55, 'INFO', 'Self-Correcting Trial 2/3 · 4 causal windows (Δt=10) · Fourier features at ω, 2ω, 3ω'),
      L(4, 0.7, 'OK', 'Trial 2/3 · Rel L2 2.84e-1 → 1.18e-2 · β ensemble collapses to unimodal 1.49'),
      L(5, 0.75, 'CRITIC', 'Residual floor 4e-4 · schedule L-BFGS fine-tune (2000 it)'),
      L(6, 0.88, 'INFO', 'Self-Correcting Trial 3/3 · L-BFGS …'),
      L(6, 0.95, 'OK', 'Trial 3/3 · Rel L2 5.62e-3 ✓ · β = 1.498 ± 0.012 (ensemble n=8)'),
      L(7, 0.98, 'CRITIC', 'Verification: energy-balance drift 0.21% ✓ · mean ODE residual 8.8e-5 ✓ → ACCEPT'),
      L(8, 1, 'AGENT', 'Research Report Agent · compiling single-column manuscript …'),
      L(9, 1, 'OK', 'Verified convergence · 3 trials · wall-clock 41.8s'),
    ],
    insight:
      'Long-horizon forced oscillators defeat single-window PINNs through phase drift. Time-marching windows with frequency-locked features eliminated the lag, and the ensemble collapsed the bimodal β posterior to 1.498 ± 0.012 (truth 1.500).',
    agentNotes: [
      'Identified a non-autonomous second-order ODE with mild chaotic sensitivity; estimated the Lyapunov exponent for horizon planning.',
      'Chose DOP853 truth, a time-marching PINN with spectral features, and an FNO baseline.',
      'Executes the reference integration and window-by-window PINN training, re-entered by the Critic loop.',
      'Audits phase lag, energy balance and the β posterior; prescribes windows, Fourier features and an ensemble.',
      'Explains hardening-spring response and forcing resonance using the verified recovered β.',
    ],
    fieldSpec: {
      kind: 'duffing', nx: 101, ny: 41, xRange: [0, 40], yRange: [0.25, 0.5],
      xLabel: 't', yLabel: 'γ (forcing)', uLabel: 'x(t; γ)', sliceVar: 'γ',
      sliceFractions: [0.2, 0.45, 0.7, 0.95], errGrowth: 1.6, uqRel: 0.03,
    },
    lossSpec: {
      epochs: 12000, step: 200, initial: 1.2,
      interventions: [
        { epoch: 3600, label: 'Critic: Time-marching + Fourier' },
        { epoch: 8800, label: 'Critic: Ensemble + L-BFGS' },
      ],
      plateaus: [3.7e-2, 2.0e-3, 8.8e-5],
    },
    bench: duffBench,
    ablation: duffAblation,
    report: {
      title: 'Agentic Physics-Informed Identification of Cubic Stiffness in a Forced Duffing Oscillator',
      subtitle: 'Wing Flutter & Grid Chaos Case Study · Inverse Stiffness Recovery',
      keywords: ['Duffing oscillator', 'time-marching PINN', 'wing flutter', 'ensemble UQ', 'inverse problem'],
      findings: [
        `Trajectory error drops from ${sci(b(duffBench, 'baseline').relL2)} (naive) to ${sci(b(duffBench, 'agent').relL2)} (self-corrected), a ${(b(duffBench, 'baseline').relL2! / b(duffBench, 'agent').relL2!).toFixed(0)}× gain.`,
        'The cubic stiffness β was recovered as 1.498 ± 0.012 versus ground truth 1.500.',
        `The energy-balance residual stays at ${b(duffBench, 'agent').driftPct}% across the 40-second horizon.`,
      ],
      interpretation: [
        'The Duffing oscillator with a hardening spring (β > 0) and negative linear stiffness (α < 0) exhibits double-well dynamics whose forced response is strongly sensitive to initial conditions. The naive PINN lost phase after t ≈ 15 and collapsed to a damped orbit, an unphysical outcome given sustained forcing.',
        `Time-marching windows impose causality, and ω-locked Fourier features resolve the forcing harmonic. The ensemble then collapsed a bimodal β posterior to a single mode, and the energy balance dE/dt = −δẋ² + γcos(ωt)ẋ is satisfied to ${b(duffBench, 'agent').driftPct}%.`,
        `The 1D FNO attains ${sci(b(duffBench, 'fno').relL2)} but extrapolates poorly beyond its training horizon, with ${b(duffBench, 'fno').driftPct}% energy-balance drift, so the agent-corrected PINN remains the trustworthy solver for long-horizon forced dynamics.`,
      ],
      methods:
        'Reference: DOP853 (rtol = 1e-12). PINN: 4 causal windows of Δt = 10, Fourier time features at ω, 2ω, 3ω, tanh MLP 5×64. Inverse variable β learned with ensemble of 8.',
      eqNote: 'Forced Duffing oscillator with unknown cubic stiffness β, t ∈ [0, 40].',
    },
  },
]

export const experimentById = (id: ExperimentId): Experiment =>
  experiments.find((e) => e.id === id) ?? experiments[0]

/* -------------------------------------------------------------------------- */
/*  Experiment history (dropdown)                                              */
/* -------------------------------------------------------------------------- */

export const history: HistoryEntry[] = [
  { id: 'run-0412', expId: 'burgers', when: 'Today · 13:42', status: 'self-corrected', relL2: '2.31e-3', note: '3 trials · RAR + Causal + L-BFGS' },
  { id: 'run-0411', expId: 'duffing', when: 'Today · 11:08', status: 'self-corrected', relL2: '5.62e-3', note: '3 trials · time-marching + ensemble' },
  { id: 'run-0409', expId: 'nlse', when: 'Yesterday · 18:25', status: 'verified', relL2: '1.76e-3', note: 'γ, δ jointly identified' },
  { id: 'run-0405', expId: 'allen-cahn', when: 'Yesterday · 09:51', status: 'self-corrected', relL2: '3.84e-3', note: 'periodic embedding fixed collapse' },
  { id: 'run-0398', expId: 'burgers', when: 'Mon · 16:03', status: 'failed', relL2: '4.87e-2', note: 'Trial 1 only · aborted by user' },
]

export const ablationBaseline = (rows: AblationRow[]) => rows[0].relL2
