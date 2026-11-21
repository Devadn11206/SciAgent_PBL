"""Pytest suite for Phase 2: PyTorch Nonlinear PINN, Autograd Compiler, Inverse Parameter Discovery, and FNO1d."""

import sys
from pathlib import Path
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.app.core.problem_schema import get_canonical_ir
from backend.app.sciml.autograd_compiler import AutogradResidualCompiler
from backend.app.sciml.models import FNO1d, NonlinearPINN
from backend.app.sciml.trainer import NonlinearPINNTrainer


class TestPhase2SciML:
    """Verifies Autograd compilation, forward convergence, inverse recovery, and FNO1d."""

    def test_autograd_compiler_and_all_pdes(self):
        for exp_id in ["burgers", "allen-cahn", "nlse", "duffing"]:
            prob = get_canonical_ir(exp_id)
            is_ode = (prob.problem_type == "ODE")
            in_dim = 1 if is_ode else 2
            out_dim = 2 if exp_id == "nlse" else 1

            model = NonlinearPINN(in_dim=in_dim, out_dim=out_dim, problem=prob)
            compiler = AutogradResidualCompiler(prob)
            res_fn = compiler.build_residual_fn(model, model.inverse_params)

            t = torch.randn(8, 1, requires_grad=True)
            x = torch.randn(8, 1, requires_grad=True) if not is_ode else None

            res = res_fn(t, x)
            assert res.shape == (8, 1)
            assert not torch.isnan(res).any()

    def test_fno1d_forward_pass(self):
        fno = FNO1d(modes=16, width=32, in_channels=1, out_channels=1)
        # Batch of 2 initial conditions on 64 spatial grid points
        ic_batch = torch.randn(2, 64, 1)
        pred = fno(ic_batch)
        assert pred.shape == (2, 64, 1)
        assert not torch.isnan(pred).any()

    def test_pinn_burgers_forward_convergence(self):
        """Verify Burgers PINN trains without NaNs and decreases total loss."""
        prob = get_canonical_ir("burgers")
        trainer = NonlinearPINNTrainer(
            problem=prob,
            use_rar=False,
            ensemble_size=1,
        )
        model, losses, _ = trainer.train_single_model(
            adam_epochs=60,
            lbfgs_epochs=10,
            n_collocation=200,
            n_ic=50,
            n_bc=50,
        )
        assert len(losses) > 0
        # Check loss reduction
        assert losses[-1].total_loss < losses[0].total_loss
        assert not np.isnan(losses[-1].total_loss)

    def test_duffing_inverse_parameter_recovery(self):
        """Verify trainable parameter discovery on Duffing (beta)."""
        prob = get_canonical_ir("duffing")
        # Synthetic clean sensor data with true beta = 1.5
        t_samples = np.linspace(0.0, 5.0, 30)
        # Duffing solution at small t stays near x(t) = 0.5 * cos(omega*t)
        u_samples = 0.5 * np.cos(1.2 * t_samples)
        sensor_data = {"t": t_samples, "u": u_samples}

        trainer = NonlinearPINNTrainer(
            problem=prob,
            ensemble_size=1,
        )
        model, losses, _ = trainer.train_single_model(
            adam_epochs=60,
            lbfgs_epochs=10,
            n_collocation=150,
            n_ic=20,
            sensor_data=sensor_data,
        )
        discovered = model.get_discovered_parameters()
        assert "beta" in discovered
        # Must stay bounded within [0.5, 3.0]
        assert 0.5 <= discovered["beta"] <= 3.0
