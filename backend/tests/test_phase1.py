"""Pytest test suite for Phase 1 deterministic solvers, metrics, and case studies."""

import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.app.core.problem_schema import (
    get_canonical_ir,
    CANONICAL_FACTORIES,
)
from backend.app.solvers.ode_solvers import (
    solve_duffing_system,
    rk4_fixed_step,
)
from backend.app.solvers.pde_solvers import (
    solve_viscous_burgers_cn,
    solve_allen_cahn_cn,
    solve_cubic_quintic_nlse_ssfm,
)
from backend.app.solvers.metrics import (
    compute_relative_l2_error,
    compute_linf_error,
    compute_conservation_drift,
    compute_mean_pde_residual_burgers,
    compute_mean_pde_residual_nlse,
    generate_synthetic_sensors,
)


class TestNonlinearODESolvers:
    """Tests for ODE integration (RK4, DOP853, Radau) on the Forced Duffing Oscillator."""

    def test_duffing_dop853_convergence_and_no_nans(self):
        sol = solve_duffing_system(
            t_span=(0.0, 40.0),
            x0=0.5,
            v0=0.0,
            method="DOP853",
            num_points=1000,
        )
        assert sol.success is True
        assert len(sol.t) == 1000
        assert not np.isnan(sol.x).any()
        assert not np.isnan(sol.v).any()
        # Bound check: Duffing amplitude remains bounded within [-2.5, 2.5]
        assert np.max(np.abs(sol.x)) < 2.5
        assert sol.solve_time_sec > 0.0

    def test_duffing_radau_implicit(self):
        sol = solve_duffing_system(
            t_span=(0.0, 20.0),
            x0=0.5,
            v0=0.0,
            method="Radau",
            num_points=500,
        )
        assert sol.success is True
        assert not np.isnan(sol.x).any()
        assert not np.isnan(sol.v).any()

    def test_duffing_rk4_comparison(self):
        sol_rk4 = solve_duffing_system(
            t_span=(0.0, 10.0),
            x0=0.5,
            v0=0.0,
            method="RK4",
            dt_fixed=0.005,
        )
        sol_dop = solve_duffing_system(
            t_span=(0.0, 10.0),
            x0=0.5,
            v0=0.0,
            method="DOP853",
            num_points=len(sol_rk4.t),
        )
        assert sol_rk4.success is True
        rel_err = compute_relative_l2_error(sol_rk4.x, sol_dop.x)
        # Over t in [0, 10], fine RK4 should agree closely with DOP853
        assert rel_err < 5e-3


class TestNonlinearPDESolvers:
    """Tests for Crank-Nicolson Burgers, Allen-Cahn, and SSFM NLSE."""

    def test_viscous_burgers_crank_nicolson(self):
        sol = solve_viscous_burgers_cn(
            x_span=(-1.0, 1.0),
            t_span=(0.0, 1.0),
            nx=128,
            nt=200,
            nu=0.01,
        )
        assert sol.success is True
        assert not np.isnan(sol.u).any()
        # Check boundary condition enforcement: u(-1, t) = 0 and u(1, t) = 0
        assert np.allclose(sol.u[:, 0], 0.0)
        assert np.allclose(sol.u[:, -1], 0.0)

        # Dissipation: L2 energy should decrease due to viscous damping
        drift_pct, energy = compute_conservation_drift(sol.u, sol.x, invariant_type="l2_energy")
        # Energy at t=1 must be strictly less than at t=0
        assert energy[-1] < energy[0]

        # Mean residual check
        res = compute_mean_pde_residual_burgers(sol.u, sol.x, sol.t, nu=0.01)
        assert res < 5e-2

    def test_allen_cahn_periodic_crank_nicolson(self):
        sol = solve_allen_cahn_cn(
            x_span=(-1.0, 1.0),
            t_span=(0.0, 0.5),
            nx=128,
            nt=150,
            eps=0.05,
        )
        assert sol.success is True
        assert not np.isnan(sol.u).any()
        # Verify periodicity: u(-1, t) == u(1, t)
        assert np.allclose(sol.u[:, 0], sol.u[:, -1], atol=1e-6)
        # Solution should stay within physical phase range [-1.5, 1.5]
        assert np.all(np.abs(sol.u) < 1.5)

    def test_cubic_quintic_nlse_ssfm_conservation(self):
        sol = solve_cubic_quintic_nlse_ssfm(
            x_span=(-6.0, 6.0),
            t_span=(0.0, 1.0),
            nx=256,
            nt=200,
            beta=1.0,
            gamma=1.0,
            delta=0.15,
        )
        assert sol.success is True
        assert not np.isnan(sol.u).any()

        # Conserved quantity: L2 optical power / norm: ∫ |psi|^2 dx must remain constant
        drift_pct, norm_traj = compute_conservation_drift(sol.u, sol.x, invariant_type="l2_energy")
        # Split-step FFT must conserve L2 norm to high precision (< 0.5% drift)
        assert drift_pct < 0.5, f"NLSE norm drift was {drift_pct:.4f}%, expected < 0.5%"


class TestMetricsAndSyntheticSensors:
    """Tests for metric calculations and synthetic sensor generation."""

    def test_metrics_accuracy(self):
        ref = np.array([1.0, 2.0, 3.0, 4.0])
        pred = np.array([1.01, 1.99, 3.02, 3.98])
        rel_l2 = compute_relative_l2_error(pred, ref)
        linf = compute_linf_error(pred, ref)

        assert rel_l2 < 0.01
        assert np.isclose(linf, 0.02)

    def test_synthetic_sensor_generation_pde(self, tmp_path):
        sol = solve_viscous_burgers_cn(nx=64, nt=50, nu=0.01)
        csv_file = str(tmp_path / "test_sensors.csv")
        csv_content, rows = generate_synthetic_sensors(
            solution_u=sol.u,
            x_coords=sol.x,
            t_coords=sol.t,
            num_sensors=32,
            noise_std=0.02,
            random_seed=123,
            output_filepath=csv_file,
        )
        assert len(rows) == 32
        assert "u_noisy" in rows[0]
        assert Path(csv_file).exists()
        with open(csv_file, "r") as f:
            lines = f.readlines()
        assert len(lines) == 33  # header + 32 rows

    def test_synthetic_sensor_generation_ode(self, tmp_path):
        sol = solve_duffing_system(t_span=(0.0, 10.0), num_points=100)
        csv_file = str(tmp_path / "duffing_sensors.csv")
        csv_content, rows = generate_synthetic_sensors(
            solution_u=sol.x,
            x_coords=None,
            t_coords=sol.t,
            num_sensors=20,
            noise_std=0.01,
            output_filepath=csv_file,
        )
        assert len(rows) == 20
        assert "u_noisy" in rows[0]
        assert Path(csv_file).exists()
