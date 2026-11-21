"""Error metrics, conservation audit calculators, and synthetic noisy sensor generators for SciAgent.

Provides:
- Relative L2 Error
- L_inf Max Absolute Error
- Mean PDE Residual on Collocation Points
- Conservation Law Drift (l2_energy and mass)
- Synthetic Noisy Sensor CSV Generator for inverse problems
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np


@dataclass
class MetricReport:
    relative_l2_error: float
    linf_max_error: float
    mean_residual: Optional[float]
    conservation_drift_pct: float
    is_converged: bool
    summary: str


# ---------------------------------------------------------------------------
# 1. Error Metrics
# ---------------------------------------------------------------------------

def compute_relative_l2_error(
    pred: np.ndarray,
    ref: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """Compute Relative L2 Error: ||u_pred - u_ref||_2 / (||u_ref||_2 + eps)."""
    diff_norm = np.linalg.norm(pred - ref)
    ref_norm = np.linalg.norm(ref)
    return float(diff_norm / (ref_norm + eps))


def compute_linf_error(pred: np.ndarray, ref: np.ndarray) -> float:
    """Compute L_infinity maximum pointwise error: max |u_pred - u_ref|."""
    return float(np.max(np.abs(pred - ref)))


# ---------------------------------------------------------------------------
# 2. PDE Residual Computation
# ---------------------------------------------------------------------------

def compute_mean_pde_residual_burgers(
    u: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    nu: float = 0.01,
) -> float:
    """Evaluate mean absolute residual |u_t + u*u_x - nu*u_xx| via finite differences."""
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    # Interior slices in time and space
    u_int = u[1:-1, 1:-1]

    # u_t centered
    u_t = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2.0 * dt)
    # u_x centered
    u_x = (u[1:-1, 2:] - u[1:-1, :-2]) / (2.0 * dx)
    # u_xx centered
    u_xx = (u[1:-1, 2:] - 2.0 * u_int + u[1:-1, :-2]) / (dx ** 2)

    residual = u_t + u_int * u_x - nu * u_xx
    return float(np.mean(np.abs(residual)))


def compute_mean_pde_residual_allen_cahn(
    u: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    eps: float = 0.05,
) -> float:
    """Evaluate mean absolute residual |u_t - eps^2*u_xx + (u^3 - u)|."""
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    u_int = u[1:-1, 1:-1]
    u_t = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2.0 * dt)
    u_xx = (u[1:-1, 2:] - 2.0 * u_int + u[1:-1, :-2]) / (dx ** 2)
    f = (u_int ** 3) - u_int

    residual = u_t - (eps ** 2) * u_xx + f
    return float(np.mean(np.abs(residual)))


def compute_mean_pde_residual_nlse(
    psi: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    beta: float = 1.0,
    gamma: float = 1.0,
    delta: float = 0.15,
) -> float:
    """Evaluate mean absolute residual of NLSE |i*psi_t + (beta/2)*psi_xx + gamma*|psi|^2*psi - delta*|psi|^4*psi|."""
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    psi_int = psi[1:-1, 1:-1]
    psi_t = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2.0 * dt)
    psi_xx = (psi[1:-1, 2:] - 2.0 * psi_int + psi[1:-1, :-2]) / (dx ** 2)

    intensity = np.abs(psi_int) ** 2
    nonlinear = gamma * intensity * psi_int - delta * (intensity ** 2) * psi_int

    residual = 1j * psi_t + 0.5 * beta * psi_xx + nonlinear
    return float(np.mean(np.abs(residual)))


# ---------------------------------------------------------------------------
# 3. Conservation Law Drift
# ---------------------------------------------------------------------------

def compute_conservation_drift(
    u: np.ndarray,
    x: np.ndarray,
    invariant_type: Literal["l2_energy", "mass"] = "l2_energy",
    eps: float = 1e-12,
) -> Tuple[float, np.ndarray]:
    """Compute conservation quantity at each time slice and calculate drift percentage:
        drift_pct = 100 * (max(I(t)) - min(I(t))) / (I(0) + eps)

    Invariants:
      - 'l2_energy': Integral of |u|^2 dx
      - 'mass': Integral of u dx
    """
    dx = x[1] - x[0]
    nt = u.shape[0]
    quantities = np.zeros(nt, dtype=np.float64)

    for n in range(nt):
        slice_vals = u[n, :]
        if invariant_type == "l2_energy":
            if np.iscomplexobj(slice_vals):
                integrand = np.abs(slice_vals) ** 2
            else:
                integrand = slice_vals ** 2
        elif invariant_type == "mass":
            integrand = np.real(slice_vals)
        else:
            raise ValueError(f"Unknown invariant_type: {invariant_type}")

        # Trapezoidal rule integration
        quantities[n] = np.trapezoid(integrand, dx=dx)

    q0 = quantities[0]
    max_dev = np.max(np.abs(quantities - q0))
    drift_pct = float(100.0 * max_dev / (abs(q0) + eps))
    return drift_pct, quantities


# ---------------------------------------------------------------------------
# 4. Synthetic Noisy Sensor CSV Generator
# ---------------------------------------------------------------------------

def generate_synthetic_sensors(
    solution_u: np.ndarray,
    x_coords: Optional[np.ndarray],
    t_coords: np.ndarray,
    num_sensors: int = 64,
    noise_std: float = 0.01,
    random_seed: int = 42,
    output_filepath: Optional[str] = None,
) -> Tuple[str, List[Dict[str, float]]]:
    """Sample spatio-temporal solution points, add Gaussian noise, and export to CSV.

    For PDE solutions: columns are [time, x, u_clean, u_noisy, noise_std]
    For ODE solutions: columns are [time, x_clean, x_noisy, noise_std]

    Returns:
        (csv_string_content, list_of_row_dictionaries)
    """
    rng = np.random.default_rng(random_seed)
    rows: List[Dict[str, float]] = []

    nt = len(t_coords)

    if x_coords is not None and solution_u.ndim == 2:
        # 1D+time PDE sampling
        nx = len(x_coords)
        # Randomly choose spatio-temporal indices
        chosen_t_idx = rng.integers(0, nt, size=num_sensors)
        chosen_x_idx = rng.integers(0, nx, size=num_sensors)

        for i in range(num_sensors):
            ti = chosen_t_idx[i]
            xi = chosen_x_idx[i]
            val = solution_u[ti, xi]
            clean_val = float(np.abs(val) if np.iscomplexobj(val) else val)
            noisy_val = clean_val + float(rng.normal(0.0, noise_std))

            rows.append({
                "t": float(t_coords[ti]),
                "x": float(x_coords[xi]),
                "u_clean": clean_val,
                "u_noisy": noisy_val,
                "noise_sigma": noise_std,
            })
        fieldnames = ["t", "x", "u_clean", "u_noisy", "noise_sigma"]

    else:
        # ODE sampling (1D state along time)
        chosen_t_idx = np.sort(rng.choice(nt, size=min(num_sensors, nt), replace=False))
        state_u = solution_u if solution_u.ndim == 1 else solution_u[0]

        for ti in chosen_t_idx:
            clean_val = float(state_u[ti])
            noisy_val = clean_val + float(rng.normal(0.0, noise_std))
            rows.append({
                "t": float(t_coords[ti]),
                "u_clean": clean_val,
                "u_noisy": noisy_val,
                "noise_sigma": noise_std,
            })
        fieldnames = ["t", "u_clean", "u_noisy", "noise_sigma"]

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    csv_content = output.getvalue()

    if output_filepath:
        with open(output_filepath, "w", newline="", encoding="utf-8") as f:
            f.write(csv_content)

    return csv_content, rows
