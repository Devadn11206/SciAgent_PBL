"""Deterministic numerical ODE solvers for SciAgent.

Supports RK4 (fixed step), adaptive DOP853 (explicit Runge-Kutta order 8(5,3)),
and implicit Radau IIA (order 5) for stiff nonlinear ODEs via scipy.integrate.solve_ivp.
Includes testing and phase portrait generation for the Forced Duffing Oscillator:
    x'' + delta*x' + alpha*x + beta*x^3 = gamma*cos(omega*t)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Literal, Optional, Tuple, Union

import numpy as np
from scipy.integrate import solve_ivp


ODEMethod = Literal["RK4", "DOP853", "Radau"]


@dataclass
class ODESolution:
    """Standardized solution structure for ODE integration."""
    method: str
    t: np.ndarray             # (N,) time grid
    x: np.ndarray             # (N,) state displacement x(t)
    v: np.ndarray             # (N,) velocity dx/dt(t)
    solve_time_sec: float     # wall-clock solve time
    success: bool
    status_message: str
    metadata: Dict[str, Union[float, int, str]]

    @property
    def phase_portrait(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (x, v) coordinates for phase portrait plotting."""
        return self.x, self.v


def rk4_fixed_step(
    fun: Callable[[float, np.ndarray], np.ndarray],
    t_span: Tuple[float, float],
    y0: np.ndarray,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Classic 4th-order Runge-Kutta fixed-step integrator."""
    t0, tf = t_span
    num_steps = int(np.ceil((tf - t0) / dt))
    t = np.linspace(t0, tf, num_steps + 1)
    actual_dt = t[1] - t[0]

    y = np.zeros((num_steps + 1, len(y0)), dtype=np.float64)
    y[0] = y0

    for i in range(num_steps):
        ti = t[i]
        yi = y[i]
        k1 = fun(ti, yi)
        k2 = fun(ti + 0.5 * actual_dt, yi + 0.5 * actual_dt * k1)
        k3 = fun(ti + 0.5 * actual_dt, yi + 0.5 * actual_dt * k2)
        k4 = fun(ti + actual_dt, yi + actual_dt * k3)
        y[i + 1] = yi + (actual_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return t, y


def solve_duffing_system(
    t_span: Tuple[float, float] = (0.0, 40.0),
    x0: float = 0.5,
    v0: float = 0.0,
    delta: float = 0.3,
    alpha: float = -1.0,
    beta: float = 1.5,
    gamma: float = 0.37,
    omega: float = 1.2,
    method: ODEMethod = "DOP853",
    num_points: Optional[int] = 2000,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    dt_fixed: float = 0.01,
) -> ODESolution:
    """Solve the forced Duffing oscillator:

        dx/dt = v
        dv/dt = -delta*v - alpha*x - beta*x**3 + gamma*cos(omega*t)

    Returns:
        ODESolution containing t, x, dx/dt, solve time, and metadata.
    """
    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        x, v = y[0], y[1]
        dxdt = v
        dvdt = -delta * v - alpha * x - beta * (x ** 3) + gamma * np.cos(omega * t)
        return np.array([dxdt, dvdt], dtype=np.float64)

    # Jacobian for implicit solver (Radau)
    def jac(t: float, y: np.ndarray) -> np.ndarray:
        x, _ = y[0], y[1]
        df1_dx, df1_dv = 0.0, 1.0
        df2_dx = -alpha - 3.0 * beta * (x ** 2)
        df2_dv = -delta
        return np.array([[df1_dx, df1_dv], [df2_dx, df2_dv]], dtype=np.float64)

    y0 = np.array([x0, v0], dtype=np.float64)
    start_time = time.perf_counter()

    if method == "RK4":
        t_arr, y_arr = rk4_fixed_step(rhs, t_span, y0, dt=dt_fixed)
        elapsed = time.perf_counter() - start_time
        success = not np.isnan(y_arr).any()
        message = "RK4 fixed-step completed successfully" if success else "RK4 produced NaNs"
        return ODESolution(
            method="RK4",
            t=t_arr,
            x=y_arr[:, 0],
            v=y_arr[:, 1],
            solve_time_sec=elapsed,
            success=success,
            status_message=message,
            metadata={"dt": dt_fixed, "num_steps": len(t_arr)},
        )

    # Adaptive methods via scipy.integrate.solve_ivp
    t_eval = np.linspace(t_span[0], t_span[1], num_points) if num_points else None

    if method == "DOP853":
        res = solve_ivp(
            rhs,
            t_span,
            y0,
            method="DOP853",
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
        )
    elif method == "Radau":
        res = solve_ivp(
            rhs,
            t_span,
            y0,
            method="Radau",
            t_eval=t_eval,
            jac=jac,
            rtol=rtol,
            atol=atol,
        )
    else:
        raise ValueError(f"Unsupported ODE method: {method}. Choose from RK4, DOP853, Radau.")

    elapsed = time.perf_counter() - start_time
    success = res.success and not np.isnan(res.y).any()

    return ODESolution(
        method=method,
        t=res.t,
        x=res.y[0],
        v=res.y[1],
        solve_time_sec=elapsed,
        success=success,
        status_message=res.message,
        metadata={
            "nfev": res.nfev,
            "njev": getattr(res, "njev", 0),
            "nlu": getattr(res, "nlu", 0),
            "rtol": rtol,
            "atol": atol,
        },
    )
