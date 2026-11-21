"""W3C Distributed Tracing and In-Memory Background Task Tracking for SciAgent MCP."""

from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


def generate_trace_id() -> str:
    """Generate 16-byte random hex trace ID (32 chars) conforming to W3C TraceContext."""
    return uuid.uuid4().hex


def generate_span_id() -> str:
    """Generate 8-byte random hex span ID (16 chars) conforming to W3C TraceContext."""
    return uuid.uuid4().hex[:16]


def format_traceparent(trace_id: str, span_id: str, sampled: bool = True) -> str:
    """Format W3C traceparent header: 00-{trace_id}-{span_id}-{flags}."""
    flags = "01" if sampled else "00"
    return f"00-{trace_id}-{span_id}-{flags}"


class MCPTraceSpan(BaseModel):
    span_id: str = Field(default_factory=generate_span_id)
    trace_id: str = Field(default_factory=generate_trace_id)
    parent_span_id: Optional[str] = None
    tool_name: str
    run_id: str
    artifact_uri: Optional[str] = None
    progress_pct: float = 0.0
    duration_ms: float = 0.0
    status: str = "running"  # running, completed, error, elicitation_needed
    details: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


class MCPTaskRecord(BaseModel):
    task_id: str
    run_id: str
    tool_name: str
    status: str = "running"  # running, completed, error
    progress_pct: float = 0.0
    current_epoch: int = 0
    total_epochs: int = 100
    message: str = "Initializing..."
    result_artifact_uri: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


# In-memory tracking singletons
ACTIVE_MCP_TASKS: Dict[str, MCPTaskRecord] = {}
RECENT_TRACE_SPANS: List[MCPTraceSpan] = []
PENDING_ELICITATIONS: Dict[str, Dict[str, Any]] = {}


def record_span(span: MCPTraceSpan) -> MCPTraceSpan:
    RECENT_TRACE_SPANS.append(span)
    if len(RECENT_TRACE_SPANS) > 200:
        RECENT_TRACE_SPANS.pop(0)
    return span


def get_run_spans(run_id: str) -> List[MCPTraceSpan]:
    return [s for s in RECENT_TRACE_SPANS if s.run_id == run_id]
