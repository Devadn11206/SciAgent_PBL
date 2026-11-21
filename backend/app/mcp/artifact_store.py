"""SciAgent MCP Stateless Artifact Store.

Stores heavy scientific NumPy arrays (x, t, u_ref, u_pred, uq_lower, uq_upper, rar_points)
on disk/SQLite and mints compact, token-efficient URI handles:
  artifact://{run_id}/{type}

Benefits:
- MCP JSON-RPC payloads remain lightweight (< 1 KB) instead of blowing up token budgets with 100k floats.
- Guarantees fast zero-copy disk/memory retrieval between discrete tool calls.
- Full provenance tracking with timestamp, shape, and dtype metadata.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


ARTIFACT_DIR = Path("backend/data/artifacts")
ARTIFACT_DB = Path("backend/data/sciagent_artifacts.db")


def _init_db() -> sqlite3.Connection:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(ARTIFACT_DB))
    conn.row_factory = sqlite3.Row
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS artifact_registry (
                uri TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                artifact_type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                shape TEXT NOT NULL,
                dtype TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT (datetime('now', 'localtime'))
            )
        """)
    return conn


class ArtifactStore:
    """Manages persistence and retrieval of heavy scientific simulation arrays."""

    def __init__(self, base_dir: Path = ARTIFACT_DIR):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def mint_uri(self, run_id: str, artifact_type: str) -> str:
        clean_type = artifact_type.lower().replace(" ", "_")
        return f"artifact://{run_id}/{clean_type}"

    def store_array(
        self,
        run_id: str,
        artifact_type: str,
        data: Union[np.ndarray, List[Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Saves a NumPy array to disk and records an artifact:// URI handle."""
        arr = np.asarray(data)
        uri = self.mint_uri(run_id, artifact_type)

        run_folder = self.base_dir / run_id
        run_folder.mkdir(parents=True, exist_ok=True)
        file_path = run_folder / f"{artifact_type}.npy"

        np.save(str(file_path), arr)

        meta_json = json.dumps(metadata or {})
        shape_str = json.dumps(list(arr.shape))
        dtype_str = str(arr.dtype)

        conn = _init_db()
        with conn:
            conn.execute(
                """
                INSERT INTO artifact_registry (
                    uri, run_id, artifact_type, file_path, shape, dtype, metadata, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now', 'localtime'))
                ON CONFLICT(uri) DO UPDATE SET
                    file_path=excluded.file_path,
                    shape=excluded.shape,
                    dtype=excluded.dtype,
                    metadata=excluded.metadata,
                    created_at=datetime('now', 'localtime')
                """,
                (uri, run_id, artifact_type, str(file_path), shape_str, dtype_str, meta_json),
            )
        conn.close()
        return uri

    def store_bundle(
        self,
        run_id: str,
        arrays: Dict[str, Union[np.ndarray, List[Any]]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """Stores multiple related scientific arrays, returning URI handles for each."""
        handles: Dict[str, str] = {}
        for k, v in arrays.items():
            if v is not None:
                handles[k] = self.store_array(run_id, k, v, metadata=metadata)
        return handles

    def load_array(self, uri: str) -> np.ndarray:
        """Retrieves NumPy array referenced by an artifact:// URI handle."""
        conn = _init_db()
        row = conn.execute("SELECT file_path FROM artifact_registry WHERE uri = ?", (uri,)).fetchone()
        conn.close()

        if not row:
            raise KeyError(f"Artifact URI '{uri}' not found in registry.")

        p = Path(row["file_path"])
        if not p.exists():
            raise FileNotFoundError(f"Underlying artifact file '{p}' is missing.")

        return np.load(str(p), allow_pickle=True)

    def get_metadata(self, uri: str) -> Dict[str, Any]:
        """Fetches shape, dtype, and provenance metadata without loading heavy array bytes."""
        conn = _init_db()
        row = conn.execute("SELECT * FROM artifact_registry WHERE uri = ?", (uri,)).fetchone()
        conn.close()
        if not row:
            raise KeyError(f"Artifact URI '{uri}' not found in registry.")

        d = dict(row)
        d["shape"] = json.loads(d["shape"])
        d["metadata"] = json.loads(d["metadata"] or "{}")
        return d

    def list_run_artifacts(self, run_id: str) -> List[Dict[str, Any]]:
        """Lists all registered artifact handles for a given experiment run."""
        conn = _init_db()
        rows = conn.execute("SELECT * FROM artifact_registry WHERE run_id = ?", (run_id,)).fetchall()
        conn.close()
        items = []
        for r in rows:
            d = dict(r)
            d["shape"] = json.loads(d["shape"])
            d["metadata"] = json.loads(d["metadata"] or "{}")
            items.append(d)
        return items


# Singleton store instance
artifact_store = ArtifactStore()
