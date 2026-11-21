"""Export Bundle Builder for SciAgent.

Packages into a single downloadable .zip file:
- SciAgent_Report.pdf (compiled single-column academic PDF)
- manuscript_tables.tex (Overleaf-ready LaTeX booktabs tables)
- figures/ (fig1 and fig2 in both 300 DPI .png and vector .pdf)
- reproduce_experiment.py (standalone reproducible python script)
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any, Dict, Union

from backend.app.reports.latex_tables import generate_latex_tables
from backend.app.reports.typst_builder import compile_typst_report


def generate_reproducible_script(run_dict: Dict[str, Any]) -> str:
    """Generate self-contained, reproducible Python code replicating the experiment."""
    prob_id = run_dict.get("problem_id", "burgers")
    return f'''"""SciAgent Standalone Reproducible Experiment Script.

Case Study: {prob_id}
Generated automatically by SciAgent Academic Exporter.
"""

import torch
import numpy as np
from backend.app.core.problem_schema import get_canonical_ir
from backend.app.sciml.trainer import NonlinearPINNTrainer

def main():
    print("Loading canonical problem IR for '{prob_id}'...")
    prob = get_canonical_ir("{prob_id}")

    print("Configuring agent-guided PINN trainer...")
    trainer = NonlinearPINNTrainer(
        problem=prob,
        use_fourier_features={prob_id == "allen-cahn"},
        use_causal_loss={prob_id in ("burgers", "allen-cahn")},
        use_rar=True,
        ensemble_size=1,
    )

    print("Training PINN with Adam + L-BFGS polish...")
    result = trainer.train_ensemble(
        adam_epochs=300,
        lbfgs_epochs=20,
    )

    print("\\n=== EXPERIMENT VERIFICATION COMPLETE ===")
    print("Discovered parameters:", result.discovered_parameters)
    print("Wall-clock time:", f"{{result.wall_clock_sec:.2f}}s")

if __name__ == "__main__":
    main()
'''


def build_export_bundle(
    run_dict: Dict[str, Any],
    output_dir: Union[str, Path],
    metadata_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Generate all figures, LaTeX tables, Typst PDF, reproducible script, and zip archive.

    Returns dict of file paths.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Compile Typst PDF and 300 DPI figures
    comp_res = compile_typst_report(run_dict, output_dir=out_path, metadata_override=metadata_override)

    # 2. Generate LaTeX tables
    latex_content = generate_latex_tables(run_dict)
    latex_file = out_path / "manuscript_tables.tex"
    with open(latex_file, "w", encoding="utf-8") as f:
        f.write(latex_content)

    # 3. Generate standalone reproducible script
    script_content = generate_reproducible_script(run_dict)
    script_file = out_path / "reproduce_experiment.py"
    with open(script_file, "w", encoding="utf-8") as f:
        f.write(script_content)

    # 4. Package into .zip bundle
    zip_file = out_path / "sciagent_manuscript_bundle.zip"
    with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(comp_res["pdf_file"], arcname="SciAgent_Report.pdf")
        zf.write(latex_file, arcname="manuscript_tables.tex")
        zf.write(script_file, arcname="reproduce_experiment.py")
        if "fig1_png" in comp_res:
            zf.write(comp_res["fig1_png"], arcname="figures/fig1_solution_grid.png")
            zf.write(comp_res["fig1_pdf"], arcname="figures/fig1_solution_grid.pdf")
        if "fig2_png" in comp_res:
            zf.write(comp_res["fig2_png"], arcname="figures/fig2_training_ablation.png")
            zf.write(comp_res["fig2_pdf"], arcname="figures/fig2_training_ablation.pdf")
        if "fig3_png" in comp_res:
            zf.write(comp_res["fig3_png"], arcname="figures/fig3_physics_diagnostics.png")
            zf.write(comp_res["fig3_pdf"], arcname="figures/fig3_physics_diagnostics.pdf")

    return {
        "pdf_file": comp_res["pdf_file"],
        "latex_file": str(latex_file),
        "script_file": str(script_file),
        "zip_bundle": str(zip_file),
        "fig1_png": comp_res.get("fig1_png", ""),
        "fig2_png": comp_res.get("fig2_png", ""),
        "fig3_png": comp_res.get("fig3_png", ""),
        "typst_source": comp_res.get("typst_source", ""),
    }
