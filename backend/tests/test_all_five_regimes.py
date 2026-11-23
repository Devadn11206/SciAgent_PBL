"""Full pipeline integration tests across all 5 physical regimes and custom PDEs:

1. Viscous Burgers (Advection-Diffusion PDE)
2. Allen-Cahn (Reaction-Diffusion PDE)
3. Cubic-Quintic NLSE (Complex-Valued Dispersive PDE)
4. Forced Duffing (1D Time Nonlinear ODE)
5. Korteweg-de Vries (3rd-Order Dispersive PDE)
6. Custom PDE (Fisher-KPP Reaction-Diffusion)
"""

import numpy as np
import pytest
import torch

from backend.app.core.problem_schema import (
    BoundaryCondition,
    DerivativeSpec,
    Domain1D,
    DomainSpec,
    InitialCondition,
    ParameterSpec,
    ProblemIR,
    create_allen_cahn_ir,
    create_cubic_quintic_nlse_ir,
    create_forced_duffing_ir,
    create_kdv_soliton_ir,
    create_viscous_burgers_ir,
)
from backend.app.sciml.trainer import NonlinearPINNTrainer
from backend.app.solvers.universal_solver import solve_arbitrary_pde
from backend.app.solvers.ode_solvers import solve_duffing_system
from backend.app.solvers.metrics import compute_relative_l2_error


def _eval_model_on_grid(model, prob, t_vals, x_vals, device):
    norm = prob.get_normalizer()
    t_norm = np.array([norm.t_transform.to_normalized(t) for t in t_vals], dtype=np.float32)
    model.eval()
    is_ode = (prob.problem_type == "ODE")

    with torch.no_grad():
        if not is_ode:
            x_norm = np.array([norm.x_transform.to_normalized(x) for x in x_vals], dtype=np.float32)
            T_grid, X_grid = np.meshgrid(t_norm, x_norm, indexing="ij")
            coords_flat = np.stack([T_grid.flatten(), X_grid.flatten()], axis=-1)
            coords_tensor = torch.tensor(coords_flat, dtype=torch.float32, device=device)
            out_pred = model(coords_tensor).cpu().numpy()
            u_pred = (
                np.sqrt(out_pred[:, 0] ** 2 + out_pred[:, 1] ** 2 + 1e-12)
                if prob.id == "nlse"
                else out_pred[:, 0]
            ).reshape(len(t_vals), len(x_vals))
        else:
            t_tensor = torch.tensor(t_norm[:, None], dtype=torch.float32, device=device)
            out_pred = model(t_tensor).cpu().numpy()[:, 0]
            u_pred = np.tile(out_pred[:, None], (1, len(x_vals) if x_vals is not None else 1))
    return u_pred


def test_burgers_pipeline():
    """1. Viscous Burgers: reference solver and autograd residual."""
    prob = create_viscous_burgers_ir()
    sol = solve_arbitrary_pde(prob, nx=100, nt=50)
    assert sol.success
    assert sol.u.shape == (50, 100)
    assert not np.isnan(sol.u).any()

    trainer = NonlinearPINNTrainer(problem=prob, use_fourier_features=True)
    m = trainer._create_model()
    t_test = torch.linspace(-1.0, 1.0, 20, device=trainer.device).unsqueeze(1).requires_grad_(True)
    x_test = torch.linspace(-1.0, 1.0, 20, device=trainer.device).unsqueeze(1).requires_grad_(True)
    res_fn = trainer.compiler.build_residual_fn(m, m.inverse_params)
    res = res_fn(t_test, x_test)
    assert res.shape == (20, 1)
    assert not torch.isnan(res).any()


def test_allen_cahn_pipeline():
    """2. Allen-Cahn: reference solver and autograd residual."""
    prob = create_allen_cahn_ir()
    sol = solve_arbitrary_pde(prob, nx=100, nt=50)
    assert sol.success
    assert sol.u.shape == (50, 100)
    assert not np.isnan(sol.u).any()

    trainer = NonlinearPINNTrainer(problem=prob, use_fourier_features=True)
    m = trainer._create_model()
    t_test = torch.linspace(-1.0, 1.0, 20, device=trainer.device).unsqueeze(1).requires_grad_(True)
    x_test = torch.linspace(-1.0, 1.0, 20, device=trainer.device).unsqueeze(1).requires_grad_(True)
    res_fn = trainer.compiler.build_residual_fn(m, m.inverse_params)
    res = res_fn(t_test, x_test)
    assert res.shape == (20, 1)
    assert not torch.isnan(res).any()


def test_nlse_pipeline():
    """3. Cubic-Quintic NLSE: Split-Step FFT and complex autograd residual."""
    prob = create_cubic_quintic_nlse_ir()
    sol = solve_arbitrary_pde(prob, nx=128, nt=50)
    assert sol.success
    assert sol.u.ndim == 2
    assert not np.isnan(sol.u).any()

    trainer = NonlinearPINNTrainer(problem=prob, use_fourier_features=True)
    m = trainer._create_model()
    assert m.out_dim == 2
    t_test = torch.linspace(-1.0, 1.0, 20, device=trainer.device).unsqueeze(1).requires_grad_(True)
    x_test = torch.linspace(-1.0, 1.0, 20, device=trainer.device).unsqueeze(1).requires_grad_(True)
    res_fn = trainer.compiler.build_residual_fn(m, m.inverse_params)
    res = res_fn(t_test, x_test)
    assert res.shape == (20, 1)
    assert not torch.isnan(res).any()


def test_duffing_pipeline():
    """4. Forced Duffing: ODE solver and 1D time autograd residual."""
    prob = create_forced_duffing_ir()
    sol = solve_duffing_system(t_span=(0.0, 5.0), num_points=100)
    assert len(sol.t) == 100
    assert not np.isnan(sol.x).any()

    trainer = NonlinearPINNTrainer(problem=prob, use_fourier_features=False)
    m = trainer._create_model()
    assert m.in_dim == 1
    t_test = torch.linspace(-1.0, 1.0, 20, device=trainer.device).unsqueeze(1).requires_grad_(True)
    res_fn = trainer.compiler.build_residual_fn(m, m.inverse_params)
    res = res_fn(t_test)
    assert res.shape == (20, 1)
    assert not torch.isnan(res).any()


def test_kdv_pipeline():
    """5. KdV Soliton: Pseudospectral Split-Step SSFM-RK4 and 3rd-order autograd residual."""
    prob = create_kdv_soliton_ir()
    sol = solve_arbitrary_pde(prob, nx=128, nt=50)
    assert sol.success
    assert sol.method == "Pseudospectral Split-Step (SSFM-RK4)"
    assert sol.u.shape == (50, 128)
    assert not np.isnan(sol.u).any()

    trainer = NonlinearPINNTrainer(problem=prob, use_fourier_features=True)
    m = trainer._create_model()
    t_test = torch.linspace(-1.0, 1.0, 20, device=trainer.device).unsqueeze(1).requires_grad_(True)
    x_test = torch.linspace(-1.0, 1.0, 20, device=trainer.device).unsqueeze(1).requires_grad_(True)
    res_fn = trainer.compiler.build_residual_fn(m, m.inverse_params)
    res = res_fn(t_test, x_test)
    assert res.shape == (20, 1)
    assert not torch.isnan(res).any()


def test_custom_fisher_kpp_pipeline():
    """6. Custom arbitrary equation: Fisher-KPP Reaction-Diffusion."""
    prob = ProblemIR(
        id="fisher-kpp",
        name="Fisher-KPP Reaction-Diffusion",
        problem_type="PDE",
        equation_latex=r"\frac{\partial u}{\partial t} = D\,\frac{\partial^2 u}{\partial x^2} + r\,u\,(1 - u)",
        derivatives=DerivativeSpec(spatial_order=2, time_order=1),
        domain=DomainSpec(
            x=Domain1D(min_val=-5.0, max_val=5.0),
            t=Domain1D(min_val=0.0, max_val=0.5),
        ),
        parameters=[
            ParameterSpec(name="D", symbol="D", latex_token="D", mode="fixed", value=0.1),
            ParameterSpec(name="r", symbol="r", latex_token="r", mode="fixed", value=1.0),
        ],
        initial_conditions=[
            InitialCondition(
                id="fkpp_ic",
                expression="u(x, 0) = exp(-x**2)",
                location="t = 0",
                order=0,
                variable="u",
            )
        ],
    )
    sol = solve_arbitrary_pde(prob, nx=100, nt=50)
    assert sol.success
    assert sol.u.shape == (50, 100)
    assert not np.isnan(sol.u).any()
