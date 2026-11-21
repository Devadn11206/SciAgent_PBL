"""SciAgent Standalone Reproducible Experiment Script.

Case Study: burgers
Generated automatically by SciAgent Academic Exporter.
"""

import torch
import numpy as np
from backend.app.core.problem_schema import get_canonical_ir
from backend.app.sciml.trainer import NonlinearPINNTrainer

def main():
    print("Loading canonical problem IR for 'burgers'...")
    prob = get_canonical_ir("burgers")

    print("Configuring agent-guided PINN trainer...")
    trainer = NonlinearPINNTrainer(
        problem=prob,
        use_fourier_features=False,
        use_causal_loss=True,
        use_rar=True,
        ensemble_size=1,
    )

    print("Training PINN with Adam + L-BFGS polish...")
    result = trainer.train_ensemble(
        adam_epochs=300,
        lbfgs_epochs=20,
    )

    print("\n=== EXPERIMENT VERIFICATION COMPLETE ===")
    print("Discovered parameters:", result.discovered_parameters)
    print("Wall-clock time:", f"{result.wall_clock_sec:.2f}s")

if __name__ == "__main__":
    main()
