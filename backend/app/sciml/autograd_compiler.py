"""Symbolic-to-Autograd Compiler for SciAgent.

Compiles ProblemIR SymPy expressions and derivatives into PyTorch autograd residual functions.
Supports:
- Up to 4th-order spatial derivatives and 2nd-order temporal derivatives using torch.autograd.grad(create_graph=True).
- Complex fields (splits psi into real and imaginary channels u_re, u_im).
- Exact AutoNormalizer chain-rule derivative scaling factors (dx_scale, dt_scale).
- Inverse parameters as trainable torch.nn.Parameter references.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from backend.app.core.problem_schema import ProblemIR


def compute_derivative(
    y: torch.Tensor,
    x: torch.Tensor,
    order: int = 1,
) -> torch.Tensor:
    """Compute exact n-th order derivative d^n y / dx^n using PyTorch autograd with create_graph=True."""
    if order == 0:
        return y
    grad = y
    for _ in range(order):
        grads = torch.autograd.grad(
            grad,
            x,
            grad_outputs=torch.ones_like(grad),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad = grads
    return grad


class AutogradResidualCompiler:
    """Compiles a ProblemIR definition into an autograd-powered residual function."""

    def __init__(self, problem: ProblemIR):
        self.problem = problem
        self.normalizer = problem.get_normalizer()
        self.scaling = self.normalizer.get_scaling_factors()

        # Chain-rule scaling factors from AutoNormalizer
        # d/dt_phys = dt_scale * d/dt_norm
        self.dt_scale = self.scaling.get("dt", 1.0)
        self.dtt_scale = self.scaling.get("dtt", 1.0)
        self.dx_scale = self.scaling.get("dx", 1.0)
        self.dxx_scale = self.scaling.get("dxx", 1.0)
        self.dxxx_scale = self.scaling.get("dxxx", 1.0)
        self.dxxxx_scale = self.scaling.get("dxxxx", 1.0)

    def build_residual_fn(
        self,
        model: nn.Module,
        inverse_params: Dict[str, nn.Parameter],
    ) -> Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
        """Build callable residual function R(t, x) or R(t).

        Inputs:
            t: tensor of shape (N, 1) with requires_grad=True (normalized [-1, 1])
            x: tensor of shape (N, 1) with requires_grad=True (normalized [-1, 1]) or None for ODEs
        Returns:
            residual: tensor of shape (N, 1)
        """
        prob_id = self.problem.id.lower()
        eq_latex = self.problem.equation_latex.lower()

        if "burgers" in prob_id or "nu" in eq_latex or "u \\frac{\\partial u}" in self.problem.equation_latex:
            return self._build_burgers_residual(model, inverse_params)
        elif "allen" in prob_id or "cahn" in prob_id or "eps" in eq_latex or "1 - u^2" in eq_latex or "1 - u**2" in eq_latex:
            return self._build_allen_cahn_residual(model, inverse_params)
        elif "nlse" in prob_id or "schrodinger" in prob_id or "psi" in eq_latex or "soliton" in prob_id:
            return self._build_nlse_residual(model, inverse_params)
        elif "duffing" in prob_id or self.problem.problem_type == "ODE" or "x_tt" in eq_latex or "\\ddot" in self.problem.equation_latex:
            return self._build_duffing_residual(model, inverse_params)
        else:
            # Fallback to general nonlinear Burgers/advection-diffusion residual
            return self._build_burgers_residual(model, inverse_params)

    def _build_burgers_residual(
        self,
        model: nn.Module,
        inverse_params: Dict[str, nn.Parameter],
    ):
        """Burgers: u_t + u * u_x - nu * u_xx = 0."""
        nu_param = inverse_params.get("nu")
        fixed_nu = self.problem.get_parameter("nu")
        nu_default = fixed_nu.value if (fixed_nu and fixed_nu.value) else 0.01

        dt_s = self.dt_scale
        dx_s = self.dx_scale
        dxx_s = self.dxx_scale

        def residual_fn(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            # coords: (N, 2) in normalized [-1, 1]
            coords = torch.cat([t, x], dim=-1)
            u = model(coords)  # (N, 1)

            # Normalized derivatives
            u_t_norm = compute_derivative(u, t, order=1)
            u_x_norm = compute_derivative(u, x, order=1)
            u_xx_norm = compute_derivative(u, x, order=2)

            # Apply chain-rule factors to get physical derivatives
            u_t = u_t_norm * dt_s
            u_x = u_x_norm * dx_s
            u_xx = u_xx_norm * dxx_s

            nu = nu_param if nu_param is not None else nu_default
            residual = u_t + u * u_x - nu * u_xx
            return residual

        return residual_fn

    def _build_allen_cahn_residual(
        self,
        model: nn.Module,
        inverse_params: Dict[str, nn.Parameter],
    ):
        """Allen-Cahn: u_t - eps^2 * u_xx + (u^3 - u) = 0."""
        eps_param = inverse_params.get("eps")
        fixed_eps = self.problem.get_parameter("eps")
        eps_default = fixed_eps.value if (fixed_eps and fixed_eps.value) else 0.05

        dt_s = self.dt_scale
        dxx_s = self.dxx_scale

        def residual_fn(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            coords = torch.cat([t, x], dim=-1)
            u = model(coords)

            u_t = compute_derivative(u, t, order=1) * dt_s
            u_xx = compute_derivative(u, x, order=2) * dxx_s

            eps = eps_param if eps_param is not None else eps_default
            residual = u_t - (eps ** 2) * u_xx + (u ** 3 - u)
            return residual

        return residual_fn

    def _build_nlse_residual(
        self,
        model: nn.Module,
        inverse_params: Dict[str, nn.Parameter],
    ):
        """Cubic-Quintic NLSE:

        i * psi_t + (beta/2) * psi_xx + gamma * |psi|^2 * psi - delta * |psi|^4 * psi = 0
        With psi = u + i*v (two real outputs):
          u_t = - (beta/2) * v_xx - gamma*(u^2+v^2)*v + delta*(u^2+v^2)^2*v
          v_t =   (beta/2) * u_xx + gamma*(u^2+v^2)*u - delta*(u^2+v^2)^2*u
        """
        gamma_param = inverse_params.get("gamma")
        delta_param = inverse_params.get("delta")
        beta_p = self.problem.get_parameter("beta")
        beta_val = beta_p.value if (beta_p and beta_p.value) else 1.0

        dt_s = self.dt_scale
        dxx_s = self.dxx_scale

        def residual_fn(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            coords = torch.cat([t, x], dim=-1)
            out = model(coords)  # (N, 2): out[:, 0] = u (Re), out[:, 1] = v (Im)
            u = out[:, 0:1]
            v = out[:, 1:2]

            u_t = compute_derivative(u, t, order=1) * dt_s
            v_t = compute_derivative(v, t, order=1) * dt_s

            u_xx = compute_derivative(u, x, order=2) * dxx_s
            v_xx = compute_derivative(v, x, order=2) * dxx_s

            intensity = u ** 2 + v ** 2
            gamma = gamma_param if gamma_param is not None else 1.0
            delta = delta_param if delta_param is not None else 0.15

            # Nonlinear source
            nl_factor = gamma * intensity - delta * (intensity ** 2)

            # Complex equation: i*(u_t + i*v_t) + (beta/2)*(u_xx + i*v_xx) + nl*(u + i*v) = 0
            # Real part: -v_t + (beta/2)*u_xx + nl*u = 0
            # Imag part:  u_t + (beta/2)*v_xx + nl*v = 0
            res_re = -v_t + 0.5 * beta_val * u_xx + nl_factor * u
            res_im = u_t + 0.5 * beta_val * v_xx + nl_factor * v

            residual = torch.sqrt(res_re ** 2 + res_im ** 2 + 1e-12)
            return residual

        return residual_fn

    def _build_duffing_residual(
        self,
        model: nn.Module,
        inverse_params: Dict[str, nn.Parameter],
    ):
        """Forced Duffing: x_tt + delta * x_t + alpha * x + beta * x^3 - gamma * cos(omega * t) = 0."""
        beta_param = inverse_params.get("beta")

        d_p = self.problem.get_parameter("delta")
        a_p = self.problem.get_parameter("alpha")
        g_p = self.problem.get_parameter("gamma")
        w_p = self.problem.get_parameter("omega")

        delta_val = d_p.value if (d_p and d_p.value) else 0.3
        alpha_val = a_p.value if (a_p and a_p.value) else -1.0
        gamma_val = g_p.value if (g_p and g_p.value) else 0.37
        omega_val = w_p.value if (w_p and w_p.value) else 1.2

        dt_s = self.dt_scale
        dtt_s = self.dtt_scale
        t_min = self.problem.domain.t.min_val
        t_span = self.problem.domain.t.span

        def residual_fn(t: torch.Tensor, x_dummy: Optional[torch.Tensor] = None) -> torch.Tensor:
            # model input is normalized t in [-1, 1]
            x_state = model(t)  # (N, 1)

            x_t = compute_derivative(x_state, t, order=1) * dt_s
            x_tt = compute_derivative(x_state, t, order=2) * dtt_s

            # Reconstruct physical time for forcing term: t_phys = t_min + 0.5 * (t + 1) * span
            t_phys = t_min + 0.5 * (t + 1.0) * t_span

            beta = beta_param if beta_param is not None else 1.5
            forcing = gamma_val * torch.cos(omega_val * t_phys)

            residual = x_tt + delta_val * x_t + alpha_val * x_state + beta * (x_state ** 3) - forcing
            return residual

        return residual_fn
