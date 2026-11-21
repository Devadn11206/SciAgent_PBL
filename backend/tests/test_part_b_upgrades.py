"""
Verification tests for Part B upgrades:
1. Natural-Language Equation Copilot (POST /api/problems/from-nl).
2. Sensor Noise & Sparsity Stress-Tester query params (GET /api/sensors/sample/{preset_id}).
3. Post-Run Grounded Critic Chat (POST /api/experiments/{run_id}/chat).
"""

import json
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.db.experiment_store import save_experiment_run

client = TestClient(app)


def test_copilot_from_nl_endpoint():
    """Verify natural-language description translation to ProblemIR."""
    # 1. Non-empty natural language query
    resp = client.post(
        "/api/problems/from-nl",
        json={"description": "1D viscous Burgers shock tube with weak kinematic viscosity"},
    )
    assert resp.status_code == 200
    problem = resp.json()
    assert "equation_latex" in problem
    assert problem["problem_type"] in ("ODE", "PDE")
    assert "domain" in problem
    assert "parameters" in problem

    # 2. Empty description validation
    resp_empty = client.post("/api/problems/from-nl", json={"description": ""})
    assert resp_empty.status_code == 400


def test_sensor_stress_tester_endpoint():
    """Verify sensor sampling accepts custom n_probes and noise_pct."""
    resp = client.get("/api/sensors/sample/burgers?n_probes=120&noise_pct=5.0")
    assert resp.status_code == 200
    csv_text = resp.text
    lines = [line for line in csv_text.strip().split("\n") if line]
    # Header + 120 sensor rows
    assert len(lines) == 121
    assert "x" in lines[0]
    assert "t" in lines[0]
    assert "u_noisy" in lines[0]


def test_grounded_critic_chat_endpoint():
    """Verify Critic chat assistant grounded in active SQLite run data."""
    test_run_id = "test_run_chat_partb_001"
    save_experiment_run(
        run_id=test_run_id,
        problem_id="burgers",
        status="verified",
        total_trials=3,
        discovered_parameters={"nu": 0.0104},
        ablation_history=[
            {"trial": "T1", "action": "Vanilla PINN", "diagnosis": "High gradient residual concentration", "relL2": 0.0487},
            {"trial": "T2", "action": "+ RAR + Causal", "diagnosis": "Curvature polish required", "relL2": 0.0095},
            {"trial": "T3", "action": "+ L-BFGS polish", "diagnosis": "Converged within tolerance", "relL2": 0.0018},
        ],
        benchmark_scores=[],
        trial1_metrics={"relative_l2_error": 0.0487, "conservation_drift_pct": 5.4},
        final_metrics={"relative_l2_error": 0.0018, "conservation_drift_pct": 0.042},
        insight_summary="Autonomous self-correction reduced relative L2 error by 27x across 3 trials.",
    )

    # Question 1: Trial 1 struggle
    q1_resp = client.post(
        f"/api/experiments/{test_run_id}/chat",
        json={"message": "Why did Trial 1 struggle?", "history": []},
    )
    assert q1_resp.status_code == 200
    res1 = q1_resp.json()
    assert res1["success"] is True
    assert "Trial 1" in res1["reply"] or "baseline" in res1["reply"] or "0.0487" in res1["reply"]

    # Question 2: Discovered parameter
    q2_resp = client.post(
        f"/api/experiments/{test_run_id}/chat",
        json={"message": "How reliable is the discovered parameter?", "history": []},
    )
    assert q2_resp.status_code == 200
    res2 = q2_resp.json()
    assert res2["success"] is True
    assert "nu" in res2["reply"] or "0.0104" in res2["reply"]
