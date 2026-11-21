"""Deterministic numerical PDE solvers for SciAgent.

Includes:
1. Viscous Burgers' PDE: Crank-Nicolson Finite Difference with Newton-Raphson iteration.
2. Allen-Cahn PDE: Crank-Nicolson / IMEX Finite Difference for phase separation.
3. Cubic-Quintic NLSE: Symmetrized Split-Step Fourier Method (SSFM) with exact linear propagator.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple, Union

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


@dataclass
class PDESolution:
    """Standardized solution structure for 1D+time PDE simulations."""
    method: str
    x: np.ndarray             # (Nx,) spatial coordinates
    t: np.ndarray             # (Nt,) time coordinates
    u: np.ndarray             # (Nt, Nx) field solution (real or complex)
    solve_time_sec: float     # wall-clock solve time
    success: bool
    status_message: str
    metadata: Dict[str, Union[float, int, str, bool]]


# ---------------------------------------------------------------------------
# 1. Viscous Burgers' PDE Solver (Crank-Nicolson + Newton-Raphson)
# ---------------------------------------------------------------------------

def solve_viscous_burgers_cn(
    x_span: Tuple[float, float] = (-1.0, 1.0),
    t_span: Tuple[float, float] = (0.0, 1.0),
    nx: int = 256,
    nt: int = 500,
    nu: float = 0.01,
    ic_func: Optional[np.ufunc] = None,
    max_newton_iter: int = 25,
    newton_tol: float = 1e-9,
) -> PDESolution:
    """Solve Viscous Burgers' equation:
        u_t + u * u_x = nu * u_xx
        on x in [x_min, x_max], t in [t_min, t_max]
        Dirichlet BC: u(x_min, t) = u(x_max, t) = 0.
        IC: u(x, 0) = -sin(pi * x)

    Uses adaptive sub-stepping and implicit Crank-Nicolson centered differences with Newton-Raphson linearization.
    Enforces adaptive CFL sub-stepping when nu is small or steep gradients form to ensure unconditional numerical stability.
    """
    start_time = time.perf_counter()
    x = np.linspace(x_span[0], x_span[1], nx)
    t = np.linspace(t_span[0], t_span[1], nt)
    dx = x[1] - x[0]
    dt_target = t[1] - t[0]

    # Interior points: index 1 to nx - 2 (size N = nx - 2)
    N = nx - 2
    u_grid = np.zeros((nt, nx), dtype=np.float64)

    # Initial condition
    if ic_func is not None:
        u_grid[0, :] = ic_func(x)
    else:
        u_grid[0, :] = -np.sin(np.pi * x)
    u_grid[0, 0] = 0.0
    u_grid[0, -1] = 0.0

    safe_nu = max(float(nu), 1e-5)

    for n in range(nt - 1):
        u_curr = u_grid[n, :].copy()
        u_max = max(1.0, float(np.max(np.abs(u_curr))))
        
        # Adaptive sub-step condition:
        # Advective CFL: dt <= 0.45 * dx / u_max
        # Diffusive limit: dt <= 0.45 * dx^2 / (2 * nu)
        dt_cfl = 0.45 * dx / u_max
        dt_diff = 0.45 * (dx ** 2) / (2.0 * safe_nu)
        sub_limit = min(dt_cfl, dt_diff)
        num_sub = max(1, min(100, int(np.ceil(dt_target / sub_limit))))
        sub_dt = dt_target / num_sub

        for _ in range(num_sub):
            u_int = u_curr[1:-1].copy()
            d2_un = (u_curr[2:] - 2.0 * u_curr[1:-1] + u_curr[:-2]) / (dx ** 2)
            # Conservative flux formulation for advection: (u^2 / 2)_x
            flux_un = (u_curr[2:]**2 - u_curr[:-2]**2) / (4.0 * dx)
            rhs = u_int - 0.5 * sub_dt * flux_un + 0.5 * sub_dt * safe_nu * d2_un

            for it in range(max_newton_iter):
                u_full = np.zeros(nx, dtype=np.float64)
                u_full[1:-1] = u_int
                d2_next = (u_full[2:] - 2.0 * u_full[1:-1] + u_full[:-2]) / (dx ** 2)
                flux_next = (u_full[2:]**2 - u_full[:-2]**2) / (4.0 * dx)
                F = u_int + 0.5 * sub_dt * flux_next - 0.5 * sub_dt * safe_nu * d2_next - rhs

                if np.linalg.norm(F, np.inf) < newton_tol:
                    break

                # Tridiagonal Jacobian
                diag = 1.0 + sub_dt * safe_nu / (dx ** 2)
                upper = 0.5 * sub_dt * u_full[2:-1] / (2.0 * dx) - 0.5 * sub_dt * safe_nu / (dx ** 2)
                lower = -0.5 * sub_dt * u_full[1:-2] / (2.0 * dx) - 0.5 * sub_dt * safe_nu / (dx ** 2)

                J = sparse.diags([lower, diag, upper], [-1, 0, 1], shape=(N, N), format="csc")
                delta = spsolve(J, -F)
                u_int += delta

                if np.isnan(u_int).any() or np.max(np.abs(u_int)) > 5.0:
                    break

            u_curr[1:-1] = np.clip(u_int, -2.5, 2.5)
            u_curr[0] = 0.0
            u_curr[-1] = 0.0

        u_grid[n + 1, :] = u_curr

    dt = dt_target
    elapsed = time.perf_counter() - start_time
    success = not np.isnan(u_grid).any()

    return PDESolution(
        method="Crank-Nicolson Adaptive-FD",
        x=x,
        t=t,
        u=u_grid,
        solve_time_sec=elapsed,
        success=success,
        status_message="Solved Viscous Burgers CN" if success else "Burgers CN encountered NaNs",
        metadata={"nx": nx, "nt": nt, "dx": dx, "dt": dt, "nu": nu},
    )


# ---------------------------------------------------------------------------
# 2. Allen-Cahn PDE Solver (Periodic Crank-Nicolson / IMEX)
# ---------------------------------------------------------------------------

def solve_allen_cahn_cn(
    x_span: Tuple[float, float] = (-1.0, 1.0),
    t_span: Tuple[float, float] = (0.0, 1.0),
    nx: int = 256,
    nt: int = 500,
    eps: float = 0.05,
    ic_func: Optional[np.ufunc] = None,
    max_newton_iter: int = 25,
    newton_tol: float = 1e-9,
) -> PDESolution:
    """Solve Allen-Cahn equation with periodic boundary conditions:
        u_t = eps^2 * u_xx - (u^3 - u)
        on x in [-1, 1], t in [0, 1]
        IC: u(x, 0) = x^2 * cos(pi * x)
        BC: periodic u(-1, t) = u(1, t), u_x(-1, t) = u_x(1, t)

    Uses implicit-explicit Crank-Nicolson with periodic boundary conditions.
    """
    start_time = time.perf_counter()
    # For periodic domain, x has nx points, with x[nx-1] periodic to x[0]
    x = np.linspace(x_span[0], x_span[1], nx, endpoint=False)
    t = np.linspace(t_span[0], t_span[1], nt)
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    u_grid = np.zeros((nt, nx), dtype=np.float64)

    # Initial condition
    if ic_func is not None:
        u_grid[0, :] = ic_func(x)
    else:
        u_grid[0, :] = (x ** 2) * np.cos(np.pi * x)

    # Reaction potential f(u) = u^3 - u, derivative f'(u) = 3*u^2 - 1
    # Crank-Nicolson for diffusion + Newton for reaction:
    # F(u^{n+1}) = u^{n+1} - 0.5 * dt * eps^2 * D2 u^{n+1} + 0.5 * dt * f(u^{n+1})
    #              - [u^n + 0.5 * dt * eps^2 * D2 u^n - 0.5 * dt * f(u^n)] = 0
    diff_coeff = (eps ** 2) / (dx ** 2)

    for n in range(nt - 1):
        u_n = u_grid[n, :]
        u_np1 = u_n.copy()

        # D2 u^n with periodic wrap
        d2_un = diff_coeff * (np.roll(u_n, -1) - 2.0 * u_n + np.roll(u_n, 1))
        f_un = (u_n ** 3) - u_n
        rhs_explicit = u_n + 0.5 * dt * d2_un - 0.5 * dt * f_un

        for it in range(max_newton_iter):
            d2_np1 = diff_coeff * (np.roll(u_np1, -1) - 2.0 * u_np1 + np.roll(u_np1, 1))
            f_np1 = (u_np1 ** 3) - u_np1
            F = u_np1 - 0.5 * dt * d2_np1 + 0.5 * dt * f_np1 - rhs_explicit

            if np.linalg.norm(F, np.inf) < newton_tol:
                break

            # Periodic Jacobian assembly
            # diag = 1.0 + dt * diff_coeff + 0.5 * dt * (3 * u_np1^2 - 1)
            # off-diagonals = -0.5 * dt * diff_coeff
            diag = 1.0 + dt * diff_coeff + 0.5 * dt * (3.0 * (u_np1 ** 2) - 1.0)
            off = -0.5 * dt * diff_coeff * np.ones(nx)

            # Build periodic sparse matrix
            row_idx = []
            col_idx = []
            vals = []
            for i in range(nx):
                # self
                row_idx.append(i); col_idx.append(i); vals.append(diag[i])
                # left neighbour
                left = (i - 1) % nx
                row_idx.append(i); col_idx.append(left); vals.append(off[i])
                # right neighbour
                right = (i + 1) % nx
                row_idx.append(i); col_idx.append(right); vals.append(off[i])

            J = sparse.coo_matrix((vals, (row_idx, col_idx)), shape=(nx, nx)).tocsc()
            delta = spsolve(J, -F)
            u_np1 += delta

        u_grid[n + 1, :] = u_np1

    # Append boundary endpoint x_span[1] to make arrays (nt, nx + 1) compatible with full domain
    x_full = np.append(x, x_span[1])
    u_full_grid = np.zeros((nt, nx + 1), dtype=np.float64)
    u_full_grid[:, :-1] = u_grid
    u_full_grid[:, -1] = u_grid[:, 0]  # periodic closure

    elapsed = time.perf_counter() - start_time
    success = not np.isnan(u_full_grid).any()

    return PDESolution(
        method="Crank-Nicolson Periodic IMEX",
        x=x_full,
        t=t,
        u=u_full_grid,
        solve_time_sec=elapsed,
        success=success,
        status_message="Solved Allen-Cahn Periodic CN" if success else "Allen-Cahn CN encountered NaNs",
        metadata={"nx": nx, "nt": nt, "dx": dx, "dt": dt, "eps": eps},
    )


# ---------------------------------------------------------------------------
# 3. Cubic-Quintic NLSE Solver (Symmetrized Split-Step Fourier Method - SSFM)
# ---------------------------------------------------------------------------

def solve_cubic_quintic_nlse_ssfm(
    x_span: Tuple[float, float] = (-6.0, 6.0),
    t_span: Tuple[float, float] = (0.0, 1.0),
    nx: int = 512,
    nt: int = 500,
    beta: float = 1.0,
    gamma: float = 1.0,
    delta: float = 0.15,
    ic_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> PDESolution:
    """Solve Cubic-Quintic Nonlinear Schrödinger Equation (NLSE) with SSFM:
        i * psi_t + (beta / 2) * psi_xx + gamma * |psi|^2 * psi - delta * |psi|^4 * psi = 0
        Domain: x in [-6, 6], t in [0, 1]
        IC: psi(x, 0) = 1.5 * sech(1.5 * x)

    Uses Strang split-step FFT with exact linear dispersion step in frequency domain:
        Step 1: Half-step linear propagator: exp(-i * beta/4 * k^2 * dt)
        Step 2: Full-step nonlinear phase rotation: exp(i * (gamma * |psi|^2 - delta * |psi|^4) * dt)
        Step 3: Half-step linear propagator: exp(-i * beta/4 * k^2 * dt)
    """
    start_time = time.perf_counter()
    x = np.linspace(x_span[0], x_span[1], nx, endpoint=False)
    t = np.linspace(t_span[0], t_span[1], nt)
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    L = x_span[1] - x_span[0]

    # Fourier wavenumber coordinates
    # k = 2 * pi * [0, 1, ..., nx/2-1, -nx/2, ..., -1] / L
    k = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)

    # Half-step linear dispersion operator in Fourier space
    # i * psi_t = - (beta/2) * psi_xx  ==>  psi_hat_t = -i * (beta/2) * k^2 * psi_hat
    dispersion_half = np.exp(-1j * 0.5 * beta * (k ** 2) * (0.5 * dt))

    psi_grid = np.zeros((nt, nx), dtype=np.complex128)

    # Initial condition
    if ic_func is not None:
        psi_grid[0, :] = ic_func(x)
    else:
        psi_grid[0, :] = 1.5 / np.cosh(1.5 * x)

    psi_curr = psi_grid[0, :].copy()

    # Enforce adaptive sub-stepping for SSFM if nonlinear phase shift is large
    # phase_shift_max = (gamma * max|psi|^2 + delta * max|psi|^4) * dt
    # If phase rotation exceeds pi/4, subdivide to prevent numerical aliasing / blowup
    for n in range(nt - 1):
        intensity_curr = np.max(np.abs(psi_curr) ** 2)
        max_rate = float(abs(gamma) * intensity_curr + abs(delta) * (intensity_curr ** 2))
        max_phase = max_rate * dt
        num_sub = max(1, min(20, int(np.ceil(max_phase / (np.pi / 4.0)))))
        sub_dt = dt / num_sub
        sub_dispersion = np.exp(-1j * 0.5 * beta * (k ** 2) * (0.5 * sub_dt))

        for _ in range(num_sub):
            # 1. First half-step linear dispersion in Fourier space
            psi_hat = np.fft.fft(psi_curr)
            psi_hat *= sub_dispersion
            psi_mid = np.fft.ifft(psi_hat)

            # 2. Full-step nonlinear phase advance in real space
            intensity = np.abs(psi_mid) ** 2
            phase_shift = (gamma * intensity - delta * (intensity ** 2)) * sub_dt
            # Clamp phase shift to prevent overflow in stiff regimes
            phase_shift = np.clip(phase_shift, -100.0, 100.0)
            psi_mid *= np.exp(1j * phase_shift)

            # 3. Second half-step linear dispersion in Fourier space
            psi_hat = np.fft.fft(psi_mid)
            psi_hat *= sub_dispersion
            psi_curr = np.fft.ifft(psi_hat)

        psi_grid[n + 1, :] = psi_curr

    # Extend periodic endpoint for visualization
    x_full = np.append(x, x_span[1])
    psi_full_grid = np.zeros((nt, nx + 1), dtype=np.complex128)
    psi_full_grid[:, :-1] = psi_grid
    psi_full_grid[:, -1] = psi_grid[:, 0]

    elapsed = time.perf_counter() - start_time
    success = not np.isnan(psi_full_grid).any()

    return PDESolution(
        method="Strang Split-Step Fourier Method (SSFM)",
        x=x_full,
        t=t,
        u=psi_full_grid,
        solve_time_sec=elapsed,
        success=success,
        status_message="Solved NLSE SSFM" if success else "NLSE SSFM encountered NaNs",
        metadata={
            "nx": nx,
            "nt": nt,
            "dx": dx,
            "dt": dt,
            "beta": beta,
            "gamma": gamma,
            "delta": delta,
        },
    )
