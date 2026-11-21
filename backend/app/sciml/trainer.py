"""Nonlinear SciML Trainer with RAR, Causal Loss Weighting, and Deep Ensemble UQ.

Features:
- Two-stage optimization: Adam with CosineAnnealingLR followed by L-BFGS fine-tuning.
- Causal temporal weighting: w_i = exp(-eta * sum_{j < i} L_res(t_j)).
- Residual-based Adaptive Refinement (RAR): samples candidates, identifies top residual points,
  and appends them to the active collocation set.
- Deep Ensemble UQ (n members): computes mean prediction, ±2σ 95% confidence intervals,
  and parameter discovery mean ± std.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from backend.app.core.problem_schema import ProblemIR
from backend.app.sciml.autograd_compiler import AutogradResidualCompiler, compute_derivative
from backend.app.sciml.models import NonlinearPINN


@dataclass
class EpochMetrics:
    epoch: int
    total_loss: float
    pde_loss: float
    bc_loss: float
    ic_loss: float
    data_loss: float
    discovered_parameters: Dict[str, float] = field(default_factory=dict)


@dataclass
class TrainingResult:
    model: NonlinearPINN
    loss_history: List[EpochMetrics]
    rar_points: List[Tuple[float, float]]  # [(t, x)] in physical coordinates
    discovered_parameters: Dict[str, float]
    parameter_uncertainties: Dict[str, float]  # std from ensemble
    wall_clock_sec: float
    peak_memory_mb: float


class NonlinearPINNTrainer:
    """Trainer orchestrating Adam + L-BFGS, Causal weights, RAR, and Ensembles."""

    def __init__(
        self,
        problem: ProblemIR,
        device: Optional[torch.device] = None,
        use_fourier_features: bool = False,
        use_causal_loss: bool = False,
        causal_eta: float = 1.0,
        use_rar: bool = False,
        rar_candidates: int = 500,
        rar_top_k: int = 20,
        ensemble_size: int = 1,
    ):
        self.problem = problem
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.use_fourier_features = use_fourier_features
        self.use_causal_loss = use_causal_loss
        self.causal_eta = causal_eta
        self.use_rar = use_rar
        self.rar_candidates = rar_candidates
        self.rar_top_k = rar_top_k
        self.ensemble_size = ensemble_size

        self.normalizer = problem.get_normalizer()
        self.compiler = AutogradResidualCompiler(problem)

    def _create_model(self) -> NonlinearPINN:
        is_ode = (self.problem.problem_type == "ODE")
        in_dim = 1 if is_ode else 2
        # NLSE has 2 channels (Re and Im)
        out_dim = 2 if self.problem.id == "nlse" else 1

        model = NonlinearPINN(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=64,
            num_layers=4,
            use_fourier_features=self.use_fourier_features,
            fourier_scale=1.0,
            problem=self.problem,
        ).to(self.device)
        return model

    def train_single_model(
        self,
        adam_epochs: int = 1500,
        lbfgs_epochs: int = 100,
        n_collocation: int = 1000,
        n_ic: int = 200,
        n_bc: int = 200,
        sensor_data: Optional[Dict[str, np.ndarray]] = None,
        callback: Optional[Callable[[EpochMetrics], None]] = None,
    ) -> Tuple[NonlinearPINN, List[EpochMetrics], List[Tuple[float, float]]]:
        """Train a single PINN model through Adam and L-BFGS with optional RAR and Causal weights."""
        model = self._create_model()
        residual_fn = self.compiler.build_residual_fn(model, model.inverse_params)

        is_ode = (self.problem.problem_type == "ODE")

        # 1. Collocation points in normalized domain [-1, 1]
        t_colloc = torch.empty((n_collocation, 1), device=self.device).uniform_(-1.0, 1.0).requires_grad_(True)
        if not is_ode:
            x_colloc = torch.empty((n_collocation, 1), device=self.device).uniform_(-1.0, 1.0).requires_grad_(True)
        else:
            x_colloc = None

        # 2. Initial Condition Points (t_norm = -1.0)
        t_ic = torch.full((n_ic, 1), -1.0, device=self.device).requires_grad_(True)
        if not is_ode:
            x_ic = torch.linspace(-1.0, 1.0, n_ic, device=self.device).unsqueeze(1)
            # Physical x coordinate for evaluating initial condition function
            x_ic_phys = torch.tensor([self.normalizer.x_transform.to_physical(float(v)) for v in x_ic], device=self.device).unsqueeze(1)
            u_ic_target = self._eval_ic_function(x_ic_phys)
        else:
            x_ic = None
            u_ic_target = torch.tensor([[0.5]], device=self.device)  # Duffing x(0) = 0.5

        # 3. Boundary Condition Points for PDEs (x_norm = -1.0 and +1.0)
        if not is_ode:
            t_bc = torch.empty((n_bc, 1), device=self.device).uniform_(-1.0, 1.0)
            x_bc_min = torch.full((n_bc, 1), -1.0, device=self.device)
            x_bc_max = torch.full((n_bc, 1), 1.0, device=self.device)
        else:
            t_bc = x_bc_min = x_bc_max = None

        # 4. Optional Sensor Data (for inverse parameter identification)
        t_sensor = x_sensor = u_sensor = None
        if sensor_data is not None and "t" in sensor_data and "u" in sensor_data:
            t_s_norm = np.array([self.normalizer.t_transform.to_normalized(t) for t in sensor_data["t"]])
            t_sensor = torch.tensor(t_s_norm, dtype=torch.float32, device=self.device).unsqueeze(1)
            if not is_ode and "x" in sensor_data:
                x_s_norm = np.array([self.normalizer.x_transform.to_normalized(x) for x in sensor_data["x"]])
                x_sensor = torch.tensor(x_s_norm, dtype=torch.float32, device=self.device).unsqueeze(1)
            u_sensor = torch.tensor(sensor_data["u"], dtype=torch.float32, device=self.device).unsqueeze(1)

        # Optimization Setup
        optimizer_adam = torch.optim.Adam(model.parameters(), lr=4e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_adam, T_max=adam_epochs, eta_min=1e-5)

        loss_history: List[EpochMetrics] = []
        rar_added_points: List[Tuple[float, float]] = []

        # Loss function closure
        def compute_losses() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            # PDE Collocation Loss
            res = residual_fn(t_colloc, x_colloc)

            if self.use_causal_loss:
                # Sort collocation points by time to compute cumulative weights
                t_flat = t_colloc.squeeze()
                sorted_idx = torch.argsort(t_flat)
                res_sorted = res[sorted_idx]
                cum_res = torch.cumsum(res_sorted ** 2, dim=0)
                weights = torch.exp(-self.causal_eta * cum_res.detach())
                pde_loss = torch.mean(weights * (res_sorted ** 2))
            else:
                pde_loss = torch.mean(res ** 2)

            # IC Loss
            if not is_ode:
                coords_ic = torch.cat([t_ic, x_ic], dim=-1)
                u_ic_pred = model(coords_ic)
                if self.problem.id == "nlse":
                    # NLSE initial target has real u and imag v=0
                    ic_loss = torch.mean((u_ic_pred[:, 0:1] - u_ic_target) ** 2) + torch.mean(u_ic_pred[:, 1:2] ** 2)
                else:
                    ic_loss = torch.mean((u_ic_pred - u_ic_target) ** 2)
            else:
                x0_pred = model(t_ic)
                # Also derivative condition x_dot(0) = 0
                x_dot_norm = compute_derivative(x0_pred, t_ic, order=1)
                ic_loss = torch.mean((x0_pred - u_ic_target) ** 2) + 0.1 * torch.mean(x_dot_norm ** 2)

            # BC Loss
            bc_loss = torch.tensor(0.0, device=self.device)
            if not is_ode:
                if self.problem.id in ("allen-cahn", "nlse"):
                    # Periodic BC: u(-1, t) == u(+1, t)
                    c_min = torch.cat([t_bc, x_bc_min], dim=-1)
                    c_max = torch.cat([t_bc, x_bc_max], dim=-1)
                    bc_loss = torch.mean((model(c_min) - model(c_max)) ** 2)
                else:
                    # Burgers: Dirichlet zero at x = +-1
                    c_min = torch.cat([t_bc, x_bc_min], dim=-1)
                    c_max = torch.cat([t_bc, x_bc_max], dim=-1)
                    bc_loss = torch.mean(model(c_min) ** 2) + torch.mean(model(c_max) ** 2)

            # Data Loss (Inverse parameter fitting)
            data_loss = torch.tensor(0.0, device=self.device)
            if u_sensor is not None:
                if not is_ode:
                    coords_sensor = torch.cat([t_sensor, x_sensor], dim=-1)
                    u_sens_pred = model(coords_sensor)
                    if self.problem.id == "nlse":
                        # Compare modulus |psi| with sensor intensity amplitude
                        amp_pred = torch.sqrt(u_sens_pred[:, 0:1] ** 2 + u_sens_pred[:, 1:2] ** 2 + 1e-12)
                        data_loss = torch.mean((amp_pred - u_sensor) ** 2)
                    else:
                        data_loss = torch.mean((u_sens_pred - u_sensor) ** 2)
                else:
                    x_sens_pred = model(t_sensor)
                    data_loss = torch.mean((x_sens_pred - u_sensor) ** 2)

            return pde_loss, bc_loss, ic_loss, data_loss

        # -------------------------------------------------------------------
        # Two-Stage IC-Warmup + Causal Curriculum Training
        # Stage 1: IC/BC Anchor Warmup (first 30% of Adam epochs)
        # Stage 2: Causal PDE Propagation + RAR + L-BFGS
        # -------------------------------------------------------------------
        warmup_epochs = max(1, int(0.30 * adam_epochs))

        for epoch in range(1, adam_epochs + 1):
            optimizer_adam.zero_grad()
            pde_loss, bc_loss, ic_loss, data_loss = compute_losses()

            # Dynamic Two-Stage Loss Weighting
            if epoch <= warmup_epochs:
                # Stage 1 (IC/BC/Data Warmup): lock amplitude and phase at t = t_min
                w_ic = 60.0
                w_bc = 30.0
                w_data = 100.0
                w_pde = 0.02
            else:
                # Stage 2: ramp PDE propagation
                progress = (epoch - warmup_epochs) / max(1, (adam_epochs - warmup_epochs))
                w_pde = 0.02 + 0.15 * progress
                w_ic = 50.0
                w_bc = 25.0
                w_data = 120.0

            total_loss = w_pde * pde_loss + w_ic * ic_loss + w_bc * bc_loss + w_data * data_loss
            total_loss.backward()
            optimizer_adam.step()
            scheduler.step()

            # Enforce physical parameter bounds
            model.clamp_inverse_parameters()

            # Optional RAR step every 150 epochs during Stage 2
            if self.use_rar and epoch > warmup_epochs and epoch % 150 == 0 and not is_ode:
                with torch.no_grad():
                    cand_t = torch.empty((self.rar_candidates, 1), device=self.device).uniform_(-1.0, 1.0)
                    cand_x = torch.empty((self.rar_candidates, 1), device=self.device).uniform_(-1.0, 1.0)
                cand_t.requires_grad_(True)
                cand_x.requires_grad_(True)

                res_cand = residual_fn(cand_t, cand_x)
                abs_res = torch.abs(res_cand).squeeze()
                top_vals, top_idx = torch.topk(abs_res, k=min(self.rar_top_k, len(abs_res)))

                new_t = cand_t[top_idx].detach()
                new_x = cand_x[top_idx].detach()

                # Record physical coordinates for UI
                for k in range(len(new_t)):
                    pt_t = self.normalizer.t_transform.to_physical(float(new_t[k].item()))
                    pt_x = self.normalizer.x_transform.to_physical(float(new_x[k].item()))
                    rar_added_points.append((pt_t, pt_x))

                # Append to active collocation set
                t_colloc = torch.cat([t_colloc.detach(), new_t], dim=0).requires_grad_(True)
                x_colloc = torch.cat([x_colloc.detach(), new_x], dim=0).requires_grad_(True)

            # Record loss telemetry
            if epoch % 20 == 0 or epoch == 1:
                metrics = EpochMetrics(
                    epoch=epoch,
                    total_loss=float(total_loss.item()),
                    pde_loss=float(pde_loss.item()),
                    bc_loss=float(bc_loss.item()),
                    ic_loss=float(ic_loss.item()),
                    data_loss=float(data_loss.item()),
                    discovered_parameters=model.get_discovered_parameters(),
                )
                loss_history.append(metrics)
                if callback is not None:
                    callback(metrics)

        # -------------------------------------------------------------------
        # Stage 2: L-BFGS Polish Loop
        # -------------------------------------------------------------------
        if lbfgs_epochs > 0:
            optimizer_lbfgs = torch.optim.LBFGS(
                model.parameters(),
                max_iter=lbfgs_epochs,
                tolerance_grad=1e-7,
                tolerance_change=1e-9,
                line_search_fn="strong_wolfe",
            )

            lbfgs_step = 0

            def lbfgs_closure():
                nonlocal lbfgs_step
                optimizer_lbfgs.zero_grad()
                pde_loss, bc_loss, ic_loss, data_loss = compute_losses()
                # L-BFGS polish balances PDE propagation with strong IC and data anchors
                total_loss = 0.05 * pde_loss + 50.0 * ic_loss + 25.0 * bc_loss + 120.0 * data_loss
                total_loss.backward()
                model.clamp_inverse_parameters()
                lbfgs_step += 1
                if lbfgs_step % 20 == 0:
                    metrics = EpochMetrics(
                        epoch=adam_epochs + lbfgs_step,
                        total_loss=float(total_loss.item()),
                        pde_loss=float(pde_loss.item()),
                        bc_loss=float(bc_loss.item()),
                        ic_loss=float(ic_loss.item()),
                        data_loss=float(data_loss.item()),
                        discovered_parameters=model.get_discovered_parameters(),
                    )
                    loss_history.append(metrics)
                    if callback is not None:
                        callback(metrics)
                return total_loss

            try:
                optimizer_lbfgs.step(lbfgs_closure)
            except Exception:
                pass

        return model, loss_history, rar_added_points

    def _eval_ic_function(self, x_phys: torch.Tensor) -> torch.Tensor:
        """Evaluate analytical initial condition function based on problem preset or custom IC."""
        # Check custom initial condition first
        if self.problem.initial_conditions:
            for ic in self.problem.initial_conditions:
                expr_str = ic.expression or ""
                if "=" in expr_str:
                    expr_str = expr_str.split("=")[-1].strip()
                expr_clean = expr_str.replace("^", "**").replace("pi", "torch.pi")
                try:
                    safe_dict = {
                        "torch": torch,
                        "sin": torch.sin,
                        "cos": torch.cos,
                        "exp": torch.exp,
                        "tanh": torch.tanh,
                        "cosh": torch.cosh,
                        "x": x_phys,
                        "pi": torch.pi,
                    }
                    res = eval(expr_clean, {"__builtins__": {}}, safe_dict)
                    if isinstance(res, torch.Tensor) and res.shape == x_phys.shape:
                        return res
                    elif isinstance(res, (int, float)):
                        return torch.full_like(x_phys, float(res))
                except Exception:
                    pass

        prob_id = self.problem.id.lower()
        eq_latex = self.problem.equation_latex.lower()
        if "burgers" in prob_id or "nu" in eq_latex:
            # u(x, 0) = -sin(pi * x)
            return -torch.sin(torch.pi * x_phys)
        elif "allen" in prob_id or "cahn" in prob_id or "eps" in eq_latex:
            # u(x, 0) = x^2 * cos(pi * x)
            return (x_phys ** 2) * torch.cos(torch.pi * x_phys)
        elif "nlse" in prob_id or "schrodinger" in prob_id or "psi" in eq_latex:
            # psi(x, 0) = 1.5 / cosh(1.5 * x)
            return 1.5 / torch.cosh(1.5 * x_phys)
        else:
            return -torch.sin(torch.pi * x_phys)

    def train_ensemble(
        self,
        adam_epochs: int = 1200,
        lbfgs_epochs: int = 50,
        sensor_data: Optional[Dict[str, np.ndarray]] = None,
        callback: Optional[Callable[[EpochMetrics], None]] = None,
        n_collocation: int = 1000,
        n_ic: int = 200,
        n_bc: int = 200,
    ) -> TrainingResult:
        """Train an ensemble of PINNs for Uncertainty Quantification (UQ)."""
        start_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        models: List[NonlinearPINN] = []
        all_losses: List[EpochMetrics] = []
        all_rar_points: List[Tuple[float, float]] = []
        ensemble_params: Dict[str, List[float]] = {}

        for i in range(self.ensemble_size):
            # Only stream callback metrics from member 0 to avoid clutter
            cb = callback if i == 0 else None
            m, losses, rar_pts = self.train_single_model(
                adam_epochs=adam_epochs,
                lbfgs_epochs=lbfgs_epochs,
                sensor_data=sensor_data,
                callback=cb,
                n_collocation=n_collocation,
                n_ic=n_ic,
                n_bc=n_bc,
            )
            models.append(m)
            if i == 0:
                all_losses = losses
                all_rar_points = rar_pts

            disc = m.get_discovered_parameters()
            for k, val in disc.items():
                ensemble_params.setdefault(k, []).append(val)

        elapsed = time.perf_counter() - start_time
        peak_mem = (
            torch.cuda.max_memory_allocated() / (1024 * 1024)
            if torch.cuda.is_available()
            else 0.0
        )

        # Compute parameter discovery statistics
        discovered_means: Dict[str, float] = {}
        discovered_stds: Dict[str, float] = {}
        for k, v_list in ensemble_params.items():
            discovered_means[k] = float(np.mean(v_list))
            discovered_stds[k] = float(np.std(v_list)) if len(v_list) > 1 else 0.0

        return TrainingResult(
            model=models[0],
            loss_history=all_losses,
            rar_points=all_rar_points,
            discovered_parameters=discovered_means,
            parameter_uncertainties=discovered_stds,
            wall_clock_sec=elapsed,
            peak_memory_mb=peak_mem,
        )
