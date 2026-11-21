"""SQLite Experiment & Checkpoint Store for SciAgent.

Persists:
- Experiment run configurations, logs, and ablation histories.
- Trial 1 vs. Final Trial metrics for comparison.
- Solution heatmaps and time-slice profiles.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


DB_DIR = Path("backend/data")
DB_PATH = DB_DIR / "sciagent_checkpoints.db"


def get_db_connection() -> sqlite3.Connection:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS experiment_runs (
                id TEXT PRIMARY KEY,
                problem_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT (datetime('now', 'localtime')),
                status TEXT NOT NULL,
                total_trials INTEGER NOT NULL,
                discovered_parameters TEXT,
                ablation_history TEXT,
                benchmark_scores TEXT,
                trial1_metrics TEXT,
                final_metrics TEXT,
                insight_summary TEXT,
                raw_payload TEXT
            )
        """)
    conn.close()


def save_experiment_run(
    run_id: str,
    problem_id: str,
    status: str,
    total_trials: int,
    discovered_parameters: Dict[str, float],
    ablation_history: List[Dict[str, Any]],
    benchmark_scores: List[Dict[str, Any]],
    trial1_metrics: Dict[str, Any],
    final_metrics: Dict[str, Any],
    insight_summary: str,
    raw_payload: Optional[Dict[str, Any]] = None,
):
    init_db()
    conn = get_db_connection()
    with conn:
        conn.execute(
            """
            INSERT INTO experiment_runs (
                id, problem_id, created_at, status, total_trials,
                discovered_parameters, ablation_history,
                benchmark_scores, trial1_metrics, final_metrics,
                insight_summary, raw_payload
            ) VALUES (?, ?, datetime('now', 'localtime'), ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                problem_id=excluded.problem_id,
                created_at=datetime('now', 'localtime'),
                status=excluded.status,
                total_trials=excluded.total_trials,
                discovered_parameters=excluded.discovered_parameters,
                ablation_history=excluded.ablation_history,
                benchmark_scores=excluded.benchmark_scores,
                trial1_metrics=excluded.trial1_metrics,
                final_metrics=excluded.final_metrics,
                insight_summary=excluded.insight_summary,
                raw_payload=excluded.raw_payload
            """,
            (
                run_id,
                problem_id,
                status,
                total_trials,
                json.dumps(discovered_parameters),
                json.dumps(ablation_history),
                json.dumps(benchmark_scores),
                json.dumps(trial1_metrics),
                json.dumps(final_metrics),
                insight_summary,
                json.dumps(raw_payload or {}),
            ),
        )
    conn.close()


def get_experiment_run(run_id: str) -> Optional[Dict[str, Any]]:
    init_db()
    conn = get_db_connection()
    row = conn.execute("SELECT * FROM experiment_runs WHERE id = ?", (run_id,)).fetchone()
    conn.close()
    if not row:
        return None
    res = dict(row)
    for field in (
        "discovered_parameters",
        "ablation_history",
        "benchmark_scores",
        "trial1_metrics",
        "final_metrics",
        "raw_payload",
    ):
        if res.get(field):
            try:
                res[field] = json.loads(res[field])
            except Exception:
                pass
    return res


def list_experiment_runs(limit: int = 50) -> List[Dict[str, Any]]:
    init_db()
    conn = get_db_connection()
    rows = conn.execute(
        "SELECT id, problem_id, created_at, status, total_trials, insight_summary, final_metrics FROM experiment_runs ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    runs = []
    for r in rows:
        d = dict(r)
        if d.get("final_metrics"):
            try:
                d["final_metrics"] = json.loads(d["final_metrics"])
            except Exception:
                pass
        runs.append(d)
    return runs
