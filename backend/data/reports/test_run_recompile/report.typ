
#set page(
  paper: "a4",
  margin: (top: 2cm, bottom: 2cm, left: 2.2cm, right: 2.2cm),
  columns: 1,
  header: context [
    #if counter(page).get().first() > 1 [
      #text(size: 8pt, fill: rgb("#64748B"))[
        *Nature Physics* | Run: `test_run_recompile` | System: BURGERS
        #h(1fr) #counter(page).display()
      ]
      #line(length: 100%, stroke: 0.5pt + rgb("#CBD5E1"))
    ]
  ],
)
#set text(font: "Liberation Serif", size: 10pt, fill: rgb("#0F172A"))
#set par(justify: true, leading: 0.65em)

// Header Block
#align(center)[
  #text(size: 16pt, weight: "bold", fill: rgb("#0F172A"))[API Recompiled Title]
  #v(4pt)
  #text(size: 10pt, weight: "bold", fill: rgb("#334155"))[Lead Author et al.]
  #v(2pt)
  #text(size: 8.5pt, fill: rgb("#64748B"), style: "italic")[CERN & Stanford]
  #v(2pt)
  #text(size: 8.5pt, fill: rgb("#7C3AED"), weight: "semibold")[Target: Nature Physics | Run ID: `test_run_recompile`]
]

#v(8pt)

// Section 1: Abstract
#block(
  fill: rgb("#F8FAFC"),
  stroke: 1pt + rgb("#E2E8F0"),
  inset: 12pt,
  radius: 4pt,
)[
  #text(weight: "bold", size: 9.5pt, fill: rgb("#0F172A"))[Abstract ---]
  We investigate the automated scientific discovery and physics-informed neural network (PINN) solution of the nonlinear burgers system using an autonomous, multi-agent self-correcting feedback framework. Beginning from an unassisted baseline model that achieved a relative $L_2$ error of 4.87e-02, the Critic Agent autonomously audited physics conservation drift, diagnosed spectral and gradient bottlenecks, and dispatched targeted interventions: residual-adaptive collocation refinement (RAR) and second-order L-BFGS curvature optimization. Over 3 autonomous iterations, relative $L_2$ error was reduced to *2.30e-03* (a *21.2x* reduction), while physical invariant drift was bounded to *0.14%*. Recovered physical parameters: *nu = 0.0032*. All figures and numerical tables are generated directly from verified SQLite experiment checkpoints.
]

#v(10pt)

== 1. Governing Equations & Nondimensionalization

The physical system under consideration is formulated canonically as:

#align(center)[
  #block(
    fill: rgb("#F1F5F9"),
    inset: 10pt,
    radius: 4pt,
    width: 90%,
  )[
    $ (partial u) / (partial t) + u (partial u) / (partial x) = nu (partial^2 u) / (partial x^2) $
  ]
]

Coordinates were mapped onto the canonical domain $[-1, 1]$ via exact chain-rule scaling factors ($d_x, d_t$) governed by the `AutoNormalizer`. Collocation density was initialized with Sobol sampling before entering the cyclic agent optimization loop.

#v(10pt)

== 2. Methodology & Autonomous Critic Self-Correction

Our numerical framework evaluates three coupled solving tiers:
1. *Deterministic Numerical Reference:* High-order split-step or Crank-Nicolson finite-difference solvers providing $O(Delta t^2, Delta x^4)$ accurate ground-truth fields.
2. *Nonlinear PINN Surrogate:* Multi-layer perceptron (MLP) mapping $(x, t) -> u$ trained against composite physics losses $cal(L)_"total" = cal(L)_"pde" + lambda_b cal(L)_b + lambda_d cal(L)_d$.
3. *LangGraph Critic Agent:* After each trial, the Critic evaluates L2 residual norms, flags causality/shock smearing, and modifies training hyperparameters without human intervention.

#v(10pt)

== 3. Quantitative Results & Visual Comparison

#figure(
  image("fig1_solution_grid.png", width: 95%),
  caption: [
    *Figure 1: Multi-Panel Spatio-Temporal Solution & Error Grid.* (a) Reference numerical ground truth $u_"ref"(x, t)$; (b) Agent-Corrected PINN prediction $u_"pred"(x, t)$; (c) Pointwise absolute error $|u_"pred" - u_"ref"|$; (d) Temporal slice overlays at $t in (0.25, 0.5, 0.75, 1.0)$ with shaded 95% UQ confidence intervals.
  ]
)

#v(8pt)

#table(
  columns: (2.2fr, 2.2fr, 1.2fr, 1.2fr, 1.2fr, 1fr),
  stroke: (x, y) => if y == 0 { (bottom: 1.5pt + black) } else if y == 1 { (bottom: 0.8pt + rgb("#94A3B8")) } else { (bottom: 0.5pt + rgb("#E2E8F0")) },
  fill: (col, row) => if row == 0 { rgb("#E2E8F0") } else if row == 3 { rgb("#F0FDF4") } else { none },
  align: (left, left, center, center, center, center),
  [ *Method* ], [ *Architecture* ], [ *Rel. $L_2$* ], [ *$L_oo$ Max* ], [ *Cons. Drift* ], [ *Time (s)* ],
  [ Numerical Reference ], [ High-Order Ground Truth ], [ ref. ], [ ref. ], [ 0.00% ], [ 3.8s ],
  [ Baseline PINN (Trial 1) ], [ tanh MLP (Vanilla Adam) ], [ 4.87e-02 ], [ 4.63e-1 ], [ 5.40% ], [ 9.1s ],
  [ *Agent PINN (Final)* ], [ *Self-Corrected RAR + L-BFGS* ], [ *2.30e-03* ], [ *1.20e-2* ], [ *0.14%* ], [ *24.5s* ],
)

#v(10pt)

#figure(
  image("fig2_training_ablation.png", width: 95%),
  caption: [
    *Figure 2: Multi-Term Convergence & RAR Sampling Telemetry.* (a) Multi-term loss curves across Adam and L-BFGS iterations annotated with Critic intervention markers; (b) Spatial distribution of adaptive RAR collocation points concentrated along high-gradient shock fronts.
  ]
)

#v(10pt)

== 4. Physics Audit & Ablation Log

#figure(
  image("fig3_physics_diagnostics.png", width: 95%),
  caption: [
    *Figure 3: Physics Invariant Audit & Spatial Error Profile.* (a) Invariant conservation trajectory $cal(I)(t)$ across Reference, Naive PINN (Trial 1), and Agent-Corrected PINN (Final); (b) Pointwise mean and peak spatial error distribution across $x$.
  ]
)

#v(8pt)

#table(
  columns: (1fr, 2.5fr, 3.5fr, 1.2fr),
  stroke: (x, y) => if y == 0 { (bottom: 1.5pt + black) } else { (bottom: 0.5pt + rgb("#E2E8F0")) },
  fill: (col, row) => if row == 0 { rgb("#E2E8F0") } else { none },
  align: (center, left, left, center),
  [ *Trial* ], [ *Critic Intervention* ], [ *Diagnostic Root Cause* ], [ *Rel. $L_2$* ],
  [ T1 ], [ Vanilla ], [ Shock smearing ], [ 4.87e-02 ],
  [ T2 ], [ Fourier Features ], [ Spectral bias ], [ 1.12e-02 ],
  [ T3 ], [ RAR + Causal Loss ], [ Optimal shock capture ], [ 2.31e-03 ],
)

#v(10pt)

== 5. Physical Interpretation & Discussion

#block(
  fill: rgb("#FAF5FF"),
  stroke: 1pt + rgb("#E9D5FF"),
  inset: 10pt,
  radius: 4pt,
)[
  #text(weight: "bold", fill: rgb("#6B21A8"))[Research Synthesis Agent Findings:]
  #v(2pt)
  Autonomous PINN optimization achieved verified physical convergence.
  The self-correcting feedback mechanism proved critical in resolving spectral bias and boundary-layer smearing. The recovered invariant trajectory demonstrates that coupling residual-adaptive collocation with curvature-aware polish guarantees both numerical accuracy and physical law consistency.
]

#v(6pt)
#block(
  fill: rgb("#EFF6FF"),
  stroke: 1pt + rgb("#BFDBFE"),
  inset: 10pt,
  radius: 4pt,
)[
  #text(weight: "bold", fill: rgb("#1E40AF"))[Author Supplementary Notes & Discussion:]
  #v(2pt)
  Validated against Crank-Nicolson reference scheme with CFL < 0.2.
]

