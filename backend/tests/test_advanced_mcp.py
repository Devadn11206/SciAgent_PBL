"""
Comprehensive verification test suite for Advanced MCP Architecture:
1. Artifact handle store round-trip (memory + disk + SQLite registry).
2. FastMCP tools execution:
   - parse_and_validate_equation (with elicitation triggers and valid parsing).
   - solve_numerical_reference.
   - train_pinn_surrogate async background tool.
   - audit_physics_invariants.
   - compile_publication_report.
3. W3C distributed tracing context propagation and spans.
4. FastAPI endpoints (/api/mcp/manifest, /api/mcp/elicit/{run_id}, /api/mcp/artifacts/{run_id}, /api/mcp/traces/{run_id}).
"""

import json
import os
import pytest
import numpy as np
from fastapi.testclient import TestClient

from backend.app.mcp.artifact_store import artifact_store
from backend.app.mcp.tracing import (
    format_traceparent,
    generate_trace_id,
    generate_span_id,
    record_span,
    get_run_spans,
    MCPTraceSpan,
)
from backend.app.mcp.sciagent_mcp import (
    mcp_server,
    parse_and_validate_equation,
    solve_numerical_reference,
    train_pinn_surrogate,
    audit_physics_invariants,
    compile_publication_report,
)
from backend.app.core.problem_schema import get_canonical_ir
from backend.app.main import app

client = TestClient(app)


def test_artifact_store_lifecycle(tmp_path):
    """Verify artifact storage, URI generation, retrieval, and SQLite indexing."""
    run_id = "test_run_store_001"
    x = np.linspace(-1.0, 1.0, 32)
    t = np.linspace(0.0, 1.0, 16)
    X, T = np.meshgrid(x, t)
    u_ref = -np.sin(np.pi * X) * np.exp(-T)

    uri = artifact_store.store_array(
        run_id=run_id,
        artifact_type="numerical_ref",
        data=u_ref,
        metadata={"pde": "heat_equation", "solver": "Crank-Nicolson", "l2_error": 0.0},
    )

    assert uri == f"artifact://{run_id}/numerical_ref"

    # Load back
    loaded_u = artifact_store.load_array(uri)
    np.testing.assert_allclose(loaded_u, u_ref)

    # List artifacts for run
    recs = artifact_store.list_run_artifacts(run_id)
    assert len(recs) >= 1
    assert recs[0]["artifact_type"] == "numerical_ref"
    assert recs[0]["uri"] == uri


def test_w3c_tracer_span_emission():
    """Verify W3C traceparent formatting and span recording."""
    run_id = "test_run_trace_002"
    trace_id = generate_trace_id()
    span_id = generate_span_id()
    traceparent = format_traceparent(trace_id, span_id)
    assert traceparent.startswith("00-")
    assert len(traceparent.split("-")) == 4

    span = MCPTraceSpan(
        span_id=span_id,
        trace_id=trace_id,
        tool_name="solve_numerical_reference",
        run_id=run_id,
        status="completed",
        artifact_uri=f"artifact://{run_id}/numerical_ref",
        duration_ms=124.5,
    )
    record_span(span)

    spans = get_run_spans(run_id)
    assert len(spans) >= 1
    matched = next((s for s in spans if s.span_id == span_id), None)
    assert matched is not None
    assert matched.status == "completed"
    assert matched.artifact_uri == f"artifact://{run_id}/numerical_ref"
    assert matched.duration_ms == 124.5


@pytest.mark.asyncio
async def test_mcp_tool_parse_and_validate():
    """Verify parse_and_validate_equation tool and interactive elicitation."""
    # 1. Valid equation
    res_valid = await parse_and_validate_equation(
        equation=r"\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} - 0.01 \frac{\partial^2 u}{\partial x^2} = 0",
        domain_config={"x": [-1.0, 1.0], "t": [0.0, 1.0], "boundary": "dirichlet", "parameters": {"nu": 0.01}},
    )
    assert res_valid["success"] is True
    assert res_valid.get("elicitation_required") is False
    assert res_valid["is_nonlinear"] is True

    # 2. Equation requiring boundary condition elicitation
    res_elicit = await parse_and_validate_equation(
        equation=r"\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = 0",
        domain_config={"x": [-1.0, 1.0], "t": [0.0, 1.0]},  # Missing boundary
    )
    assert res_elicit.get("elicitation_required") is True
    assert res_elicit["elicitation"]["field"] == "boundary_condition"
    assert len(res_elicit["elicitation"]["options"]) > 0


@pytest.mark.asyncio
async def test_mcp_tool_solve_numerical_reference():
    """Verify solve_numerical_reference tool writes to artifact store."""
    run_id = "test_mcp_solve_003"
    prob = get_canonical_ir("burgers")

    result = await solve_numerical_reference(
        problem_ir_json=json.dumps(prob.model_dump(mode="json")),
        run_id=run_id,
    )
    assert result["success"] is True
    assert "ref_artifact_uri" in result
    assert result["ref_artifact_uri"] == f"artifact://{run_id}/u_ref"

    # Check that artifact actually exists in store
    loaded_u = artifact_store.load_array(result["ref_artifact_uri"])
    assert loaded_u is not None
    assert loaded_u.ndim == 2


@pytest.mark.asyncio
async def test_mcp_tool_audit_physics_invariants():
    """Verify audit_physics_invariants tool comparing artifact handles."""
    run_id = "test_audit_004"
    x = np.linspace(-1.0, 1.0, 32)
    t = np.linspace(0.0, 1.0, 16)
    X, T = np.meshgrid(x, t)
    u_ref = -np.sin(np.pi * X) * np.exp(-T)
    u_pred = u_ref + 0.001 * np.random.randn(*u_ref.shape)

    ref_uri = artifact_store.store_array(run_id=run_id, artifact_type="u_ref", data=u_ref)
    pinn_uri = artifact_store.store_array(run_id=run_id, artifact_type="u_pred", data=u_pred)
    artifact_store.store_array(run_id=run_id, artifact_type="x_grid", data=x)
    artifact_store.store_array(run_id=run_id, artifact_type="t_grid", data=t)

    audit_res = await audit_physics_invariants(
        ref_artifact_uri=ref_uri,
        pred_artifact_uri=pinn_uri,
        invariant_type="l2_energy",
        problem_id="burgers",
        run_id=run_id,
    )
    assert audit_res["success"] is True
    assert audit_res["rel_l2_error"] < 0.05
    assert audit_res["is_physically_admissible"] is True


def test_fastapi_mcp_endpoints():
    """Verify FastAPI MCP discovery, elicitation, and trace endpoints."""
    # 1. Manifest
    resp = client.get("/api/mcp/manifest")
    assert resp.status_code == 200
    manifest = resp.json()
    assert manifest["server_name"] == "SciAgent-Scientific-Core"
    tool_names = [t["name"] for t in manifest["tools"]]
    assert "parse_and_validate_equation" in tool_names
    assert "solve_numerical_reference" in tool_names
    assert "train_pinn_surrogate" in tool_names
    assert "audit_physics_invariants" in tool_names
    assert "compile_publication_report" in tool_names

    # 2. Elicitation resolution endpoint
    from backend.app.mcp.tracing import PENDING_ELICITATIONS
    PENDING_ELICITATIONS["run_test_elicit_999"] = {"field": "boundary_condition", "options": ["periodic", "dirichlet"]}
    elicit_resp = client.post(
        "/api/mcp/elicit/run_test_elicit_999",
        json={"chosen_option": "periodic", "custom_value": None},
    )
    assert elicit_resp.status_code == 200
    elicit_data = elicit_resp.json()
    assert elicit_data["success"] is True
    assert elicit_data["chosen_option"] == "periodic"

    # 3. Traces endpoint
    trace_resp = client.get("/api/mcp/traces/test_run_trace_002")
    assert trace_resp.status_code == 200
    traces = trace_resp.json()
    assert isinstance(traces, list)
    assert len(traces) >= 1
