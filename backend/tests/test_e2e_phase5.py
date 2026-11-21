"""End-to-end tests for Phase 5.

Verifies the entire pipeline:
1. Preset retrieval & SymPy parsing.
2. Synthetic sensor generation & Multipart CSV upload with validation statistics.
3. LangGraph self-correcting agent execution with SQLite persistence.
4. Publication figure creation, LaTeX table export, single-column Typst academic PDF compilation, and reproducible ZIP bundle creation.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.core.problem_schema import get_canonical_ir
from backend.app.core.symbolic_parser import parse_symbolic_equation
from backend.app.db.experiment_store import get_experiment_run, list_experiment_runs
from backend.app.main import app
from backend.app.reports.export_bundle import build_export_bundle
from backend.app.reports.latex_tables import generate_latex_tables
from backend.app.reports.pub_plots import generate_publication_figures
from backend.app.reports.typst_builder import compile_typst_report
from backend.app.solvers.metrics import generate_synthetic_sensors
from backend.app.solvers.pde_solvers import solve_viscous_burgers_cn


@pytest.fixture
def client():
    return TestClient(app)


class TestPhase5EndToEnd:
    """Complete E2E integration test suite."""

    def test_e2e_flow_preset_parse_and_sensor_upload(self, client):
        # 1. Fetch presets
        resp_presets = client.get("/api/presets")
        assert resp_presets.status_code == 200
        presets = resp_presets.json()
        assert len(presets) >= 4

        # 2. Parse equation
        burgers = next(p for p in presets if p["id"] == "burgers")
        resp_parse = client.post("/api/problems/parse", json={"equation": burgers["equation_latex"]})
        assert resp_parse.status_code == 200
        parsed = resp_parse.json()
        assert "regimes" in parsed or "primary_regime" in parsed
        assert parsed["is_nonlinear"] is True
        assert parsed["well_posedness"]["is_well_posed"] is True

        # 3. Sample synthetic sensors
        resp_sample = client.get("/api/sensors/sample/burgers")
        assert resp_sample.status_code == 200
        csv_text = resp_sample.text
        lines = csv_text.strip().splitlines()
        assert len(lines) >= 60
        assert "t,x,u_clean" in lines[0] or "t,x,u" in lines[0]

        # 4. Upload multipart CSV sensor file
        files = {
            "file": ("test_sensors.csv", io.BytesIO(csv_text.encode("utf-8")), "text/csv")
        }
        resp_upload = client.post("/api/sensors/upload", files=files)
        assert resp_upload.status_code == 200
        up_data = resp_upload.json()
        assert up_data["success"] is True
        assert up_data["kind"] == "PDE"
        assert up_data["row_count"] >= 60
        assert "statistics" in up_data
        assert up_data["statistics"]["t_min"] >= 0.0

    def test_e2e_langgraph_agent_and_sqlite(self, client):
        # 1. Trigger agent run with 1 trial for speed
        prob = get_canonical_ir("burgers")
        resp_run = client.post(
            "/api/agents/run",
            json={"problem": prob.model_dump(), "max_trials": 1, "run_id": "test_e2e_phase5_run"},
        )
        assert resp_run.status_code == 200
        data = resp_run.json()
        assert data["success"] is True
        assert data["run_id"] == "test_e2e_phase5_run"
        assert data["status"] == "COMPLETED"

        # 2. Verify SQLite persistence
        db_rec = get_experiment_run("test_e2e_phase5_run")
        assert db_rec is not None
        assert db_rec["problem_id"] == "burgers"
        assert "final_metrics" in db_rec or "benchmark_scores" in db_rec

    def test_e2e_report_artifacts_generation(self, client, tmp_path):
        # 1. Call reports generate endpoint
        resp_rep = client.post("/api/reports/test_e2e_phase5_run/generate")
        assert resp_rep.status_code == 200
        rep_data = resp_rep.json()
        assert rep_data["success"] is True
        assert "pdf_path" in rep_data
        assert "bundle_path" in rep_data

        # 2. Verify PDF download
        resp_pdf = client.get("/api/reports/test_e2e_phase5_run/pdf")
        assert resp_pdf.status_code == 200
        assert resp_pdf.headers["content-type"] == "application/pdf"
        assert len(resp_pdf.content) > 10000

        # 3. Verify ZIP bundle download
        resp_zip = client.get("/api/reports/test_e2e_phase5_run/bundle")
        assert resp_zip.status_code == 200
        assert resp_zip.headers["content-type"] == "application/zip"
        assert len(resp_zip.content) > 50000

        # 4. Verify standalone script download
        resp_py = client.get("/api/reports/test_e2e_phase5_run/script")
        assert resp_py.status_code == 200
        assert "NonlinearPINN" in resp_py.text
