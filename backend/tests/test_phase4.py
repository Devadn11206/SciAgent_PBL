"""Pytest suite for Phase 4: 300 DPI figures, LaTeX booktabs tables, Typst single-column PDF, and .zip bundle."""

import sys
from pathlib import Path
import zipfile
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.app.reports.export_bundle import build_export_bundle
from backend.app.reports.latex_tables import generate_latex_tables
from backend.app.reports.pub_plots import generate_publication_figures
from backend.app.reports.typst_builder import compile_typst_report


class TestPhase4Reports:
    """Verifies publication-grade report generators and compilers."""

    @pytest.fixture
    def sample_run_data(self):
        return {
            "id": "test_run_phase4_sample",
            "problem_id": "burgers",
            "insight_summary": "Critic self-correction cut shock layer error 7x.",
            "final_metrics": {
                "rel_l2": 2.31e-3,
                "drift_pct": 0.14,
                "wall_s": 24.5,
            },
            "trial1_metrics": {
                "rel_l2": 4.87e-2,
                "drift_pct": 2.9,
                "wall_s": 9.1,
            },
            "discovered_parameters": {"nu": 0.01003},
            "benchmark_scores": [
                {"method": "Numerical Truth", "detail": "CN", "kind": "reference", "relL2": None, "linf": None, "meanRes": 4.1e-8, "driftPct": 0.002, "wallS": 3.8, "memMB": 212},
                {"method": "Baseline PINN", "detail": "tanh", "kind": "baseline", "relL2": 4.87e-2, "linf": 3.41e-1, "meanRes": 8.6e-3, "driftPct": 2.90, "wallS": 9.1, "memMB": 640},
                {"method": "Agent PINN", "detail": "RAR+LBFGS", "kind": "agent", "relL2": 2.31e-3, "linf": 1.87e-2, "meanRes": 4.2e-5, "driftPct": 0.14, "wallS": 24.5, "memMB": 702},
            ],
            "ablation_history": [
                {"trial": "T1", "action": "Vanilla", "diagnosis": "Shock smearing", "relL2": 0.0487},
                {"trial": "T2", "action": "+ RAR", "diagnosis": "Interface spikes", "relL2": 0.0069},
                {"trial": "T3", "action": "+ L-BFGS", "diagnosis": "Floor drop", "relL2": 0.00231},
            ],
            "raw_payload": {},
        }

    def test_300dpi_publication_figures(self, tmp_path, sample_run_data):
        figs = generate_publication_figures(sample_run_data, output_dir=tmp_path)
        assert Path(figs["fig1_png"]).exists()
        assert Path(figs["fig1_pdf"]).exists()
        assert Path(figs["fig2_png"]).exists()
        assert Path(figs["fig2_pdf"]).exists()
        # Verify non-trivial file size (> 10 KB)
        assert Path(figs["fig1_png"]).stat().st_size > 10000

    def test_latex_tables_syntax(self, sample_run_data):
        tex = generate_latex_tables(sample_run_data)
        assert "\\toprule" in tex
        assert "\\midrule" in tex
        assert "\\bottomrule" in tex
        assert "Multi-Method Quantitative Benchmark Scorecard" in tex
        assert "Agentic Self-Correction & Ablation Audit Log" in tex

    def test_typst_single_column_pdf_compilation(self, tmp_path, sample_run_data):
        comp = compile_typst_report(sample_run_data, output_dir=tmp_path)
        pdf_path = Path(comp["pdf_file"])
        assert pdf_path.exists()
        # Verify valid non-empty PDF file
        assert pdf_path.stat().st_size > 50000
        with open(pdf_path, "rb") as f:
            header = f.read(4)
            assert header == b"%PDF"

    def test_export_bundle_zip_integrity(self, tmp_path, sample_run_data):
        bundle = build_export_bundle(sample_run_data, output_dir=tmp_path)
        zip_path = Path(bundle["zip_bundle"])
        assert zip_path.exists()
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            assert "SciAgent_Report.pdf" in names
            assert "manuscript_tables.tex" in names
            assert "reproduce_experiment.py" in names
            assert "figures/fig1_solution_grid.png" in names
            assert "figures/fig2_training_ablation.png" in names
