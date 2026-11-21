"""Unit test verifying the upgraded Single-Column Academic Report generation:
- fig3_physics_diagnostics generation (300 DPI, conservation trajectory & spatial error)
- Typst 0.15 strict single-column layout compilation
- Recompilation endpoint with custom metadata & discussion notes
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.reports.pub_plots import generate_publication_figures
from backend.app.reports.typst_builder import compile_typst_report

client = TestClient(app)

def test_publication_figures_generation_including_fig3(tmp_path: Path):
    """Test generating 300 DPI Figures including Figure 3 physics diagnostics."""
    data = {
        "problem_id": "burgers",
        "problem_name": "Viscous Burgers' Equation",
        "equation": r"\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}",
        "equation_latex": r"\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}",
        "final_metrics": {"rel_l2": 0.0018, "drift_pct": 0.03, "wall_s": 14.2},
        "trial1_metrics": {"relative_l2_error": "6.35e-2", "conservation_drift_pct": 1.45},
        "ablation_history": [
            {"trial": "T1", "action": "Vanilla PINN", "diagnosis": "Shock smearing", "relative_l2_error": 0.0635, "verdict": "baseline"},
            {"trial": "T2", "action": "+ RAR + Causal", "diagnosis": "Optimizer stalled", "relative_l2_error": 0.0082, "verdict": "accepted"},
            {"trial": "T3", "action": "+ L-BFGS polish", "diagnosis": "Converged", "relative_l2_error": 0.0018, "verdict": "final"},
        ],
        "benchmark_scores": [],
    }

    figs = generate_publication_figures(data, output_dir=tmp_path)
    assert "fig1_png" in figs
    assert "fig2_png" in figs
    assert "fig3_png" in figs
    assert "fig3_pdf" in figs
    assert Path(figs["fig3_png"]).exists()
    assert Path(figs["fig3_png"]).stat().st_size > 10000

def test_typst_single_column_compilation(tmp_path: Path):
    """Test compiling the full single-column Typst manuscript."""
    data = {
        "problem_id": "burgers",
        "problem_name": "Viscous Burgers' Equation",
        "equation_latex": r"\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}",
        "final_metrics": {"rel_l2": 0.0018, "drift_pct": 0.03, "wall_s": 14.2},
        "trial1_metrics": {"relative_l2_error": "6.35e-2", "conservation_drift_pct": 1.45},
        "ablation_history": [
            {"trial": "T1", "action": "Vanilla PINN", "diagnosis": "Shock smearing", "relative_l2_error": 0.0635, "verdict": "baseline"},
            {"trial": "T2", "action": "+ RAR + Causal", "diagnosis": "Optimizer stalled", "relative_l2_error": 0.0082, "verdict": "accepted"},
            {"trial": "T3", "action": "+ L-BFGS polish", "diagnosis": "Converged", "relative_l2_error": 0.0018, "verdict": "final"},
        ],
        "benchmark_scores": [],
    }

    figs = generate_publication_figures(data, output_dir=tmp_path)
    results = compile_typst_report(
        data,
        output_dir=tmp_path,
        metadata_override={
            "title": "Custom Test Manuscript on Viscous Burgers Dynamics",
            "authors": "Dr. Scientist and SciAgent Core",
            "affiliation": "Physical Artificial Intelligence Institute",
            "custom_notes": "All numerical tests confirmed energy conservation under Dirichlet bounds.",
        }
    )

    pdf_path = Path(results["pdf_file"])
    typ_path = Path(results["typst_file"])

    assert typ_path.exists()
    typ_content = typ_path.read_text(encoding="utf-8")
    assert "columns: 1" in typ_content
    assert "Custom Test Manuscript on Viscous Burgers Dynamics" in typ_content
    assert "Physical Artificial Intelligence Institute" in typ_content
    assert "All numerical tests confirmed energy conservation" in typ_content

    # If typst binary is compiled
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 5000

def test_recompile_endpoint_api():
    """Test POST /api/reports/{run_id}/recompile endpoint."""
    resp = client.post(
        "/api/reports/test_run_recompile/recompile",
        json={
            "title": "API Recompiled Title",
            "authors": "Lead Author et al.",
            "affiliation": "CERN & Stanford",
            "journal": "Nature Physics",
            "custom_notes": "Validated against Crank-Nicolson reference scheme with CFL < 0.2.",
        }
    )
    assert resp.status_code == 200
    res = resp.json()
    assert res["success"] is True
    assert "typst_source" in res
    assert "API Recompiled Title" in res["typst_source"]
    assert "CERN & Stanford" in res["typst_source"]
    assert "latex_tables" in res
