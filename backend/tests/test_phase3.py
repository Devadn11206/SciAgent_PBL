"""Pytest suite for Phase 3: LangGraph 5-node cyclic workflow, SQLite checkpointing, and Critic self-correction."""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.app.core.problem_schema import get_canonical_ir
from backend.app.agents.graph import create_sciagent_graph, SciAgentState
from backend.app.db.experiment_store import get_experiment_run, list_experiment_runs
from backend.app.agents.llm_client import OllamaClient, ProblemDiagnosis, CriticVerdict


class TestPhase3LangGraph:
    """Tests for LangGraph cyclic execution and SQLite persistence."""

    def test_ollama_client_health_and_fallback(self):
        client = OllamaClient()
        is_online, model = client.check_health()
        print(f"Ollama online: {is_online}, model: {model}")
        assert isinstance(is_online, bool)

    def test_sqlite_persistence(self):
        run_id = "test_run_persist_123"
        from backend.app.db.experiment_store import save_experiment_run

        save_experiment_run(
            run_id=run_id,
            problem_id="burgers",
            status="COMPLETED",
            total_trials=2,
            discovered_parameters={"nu": 0.0102},
            ablation_history=[{"trial": "T1", "relL2": 0.04}, {"trial": "T2", "relL2": 0.005}],
            benchmark_scores=[{"method": "agent", "relL2": 0.005}],
            trial1_metrics={"rel_l2": 0.04},
            final_metrics={"rel_l2": 0.005},
            insight_summary="Self-correction cut error 8x.",
        )

        res = get_experiment_run(run_id)
        assert res is not None
        assert res["id"] == run_id
        assert res["total_trials"] == 2
        assert res["final_metrics"]["rel_l2"] == 0.005

    def test_langgraph_cyclic_execution_flow(self):
        """Run the 5-node graph with max_trials=2 to verify self-correction loop and transition logic."""
        prob = get_canonical_ir("burgers")
        graph = create_sciagent_graph()

        initial_state: SciAgentState = {
            "run_id": "test_graph_cyclic_456",
            "problem": prob.model_dump(),
            "current_trial": 1,
            "max_trials": 2,
            "stage": 0,
            "active_node": "Problem Analysis Agent",
            "diagnosis": None,
            "current_plan": None,
            "trial_results": [],
            "critic_verdict": None,
            "ablation_history": [],
            "research_summary": None,
            "logs": [],
            "is_converged": False,
            "final_output": None,
        }

        # Execute through the graph
        final_state = graph.invoke(initial_state)

        assert final_state["stage"] == 9
        assert len(final_state["trial_results"]) >= 1
        assert len(final_state["ablation_history"]) >= 1
        assert final_state["research_summary"] is not None

        # Verify record was stored in SQLite
        db_run = get_experiment_run("test_graph_cyclic_456")
        assert db_run is not None
        assert db_run["status"] == "COMPLETED"
