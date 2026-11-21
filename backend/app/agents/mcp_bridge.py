"""MCP Client Bridge with W3C Distributed Trace Propagation.

Connects LangGraph nodes and FastAPI endpoints directly to FastMCP tools,
attaching W3C traceparent headers and streaming structured MCP_TRACE events.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Callable, Coroutine, Dict, List, Optional
from backend.app.mcp.sciagent_mcp import mcp_server
from backend.app.mcp.tracing import (
    MCPTraceSpan,
    format_traceparent,
    generate_span_id,
    generate_trace_id,
    record_span,
)


class MCPClientBridge:
    """Dispatches calls to FastMCP tools while propagating W3C trace context."""

    def __init__(self, run_id: str, trace_id: Optional[str] = None):
        self.run_id = run_id
        self.trace_id = trace_id or generate_trace_id()
        self.active_span_id: Optional[str] = None
        self._event_subscribers: List[Callable[[Dict[str, Any]], Coroutine[Any, Any, None]]] = []

    def subscribe(self, callback: Callable[[Dict[str, Any]], Coroutine[Any, Any, None]]):
        self._event_subscribers.append(callback)

    async def _emit_event(self, event: Dict[str, Any]):
        for sub in list(self._event_subscribers):
            try:
                await sub(event)
            except Exception:
                pass

    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        parent_span_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute FastMCP tool, log W3C span, and broadcast trace event."""
        span_id = generate_span_id()
        parent_id = parent_span_id or self.active_span_id
        t0 = time.time()

        traceparent = format_traceparent(self.trace_id, span_id)

        # Notify start
        start_event = {
            "type": "mcp_trace",
            "run_id": self.run_id,
            "span_id": span_id,
            "parent_span_id": parent_id,
            "trace_id": self.trace_id,
            "traceparent": traceparent,
            "tool_name": tool_name,
            "status": "started",
            "progress_pct": 0.0,
            "timestamp": t0,
        }
        await self._emit_event(start_event)

        # Inject trace metadata into tool kwargs
        kwargs = dict(arguments)
        kwargs["trace_id"] = self.trace_id
        kwargs["parent_span_id"] = parent_id
        if "run_id" not in kwargs:
            kwargs["run_id"] = self.run_id

        try:
            # Call FastMCP tool directly
            res = await mcp_server.call_tool(tool_name, kwargs)

            # Unpack structured result or content
            unpacked: Dict[str, Any] = {}
            if hasattr(res, "structured_content") and res.structured_content:
                if isinstance(res.structured_content, dict) and "result" in res.structured_content:
                    unpacked = res.structured_content["result"]
                else:
                    unpacked = res.structured_content
            elif hasattr(res, "content") and res.content:
                for c in res.content:
                    if hasattr(c, "text"):
                        try:
                            unpacked = json.loads(c.text)
                        except Exception:
                            unpacked = {"text": c.text}

            duration_ms = (time.time() - t0) * 1000.0
            artifact_uri = unpacked.get("ref_artifact_uri") or unpacked.get("pred_artifact_uri")

            complete_span = MCPTraceSpan(
                span_id=span_id,
                trace_id=self.trace_id,
                parent_span_id=parent_id,
                tool_name=tool_name,
                run_id=self.run_id,
                artifact_uri=artifact_uri,
                progress_pct=100.0,
                duration_ms=duration_ms,
                status="elicitation_needed" if unpacked.get("elicitation_required") else "completed",
                details=unpacked,
                timestamp=time.time(),
            )
            record_span(complete_span)

            end_event = {
                "type": "mcp_trace",
                "run_id": self.run_id,
                "span_id": span_id,
                "parent_span_id": parent_id,
                "trace_id": self.trace_id,
                "traceparent": traceparent,
                "tool_name": tool_name,
                "artifact_uri": artifact_uri,
                "status": complete_span.status,
                "duration_ms": duration_ms,
                "progress_pct": 100.0,
                "details": unpacked,
                "timestamp": time.time(),
            }
            await self._emit_event(end_event)
            return unpacked

        except Exception as e:
            duration_ms = (time.time() - t0) * 1000.0
            err_span = MCPTraceSpan(
                span_id=span_id,
                trace_id=self.trace_id,
                parent_span_id=parent_id,
                tool_name=tool_name,
                run_id=self.run_id,
                duration_ms=duration_ms,
                status="error",
                details={"error": str(e)},
                timestamp=time.time(),
            )
            record_span(err_span)

            err_event = {
                "type": "mcp_trace",
                "run_id": self.run_id,
                "span_id": span_id,
                "parent_span_id": parent_id,
                "trace_id": self.trace_id,
                "traceparent": traceparent,
                "tool_name": tool_name,
                "status": "error",
                "duration_ms": duration_ms,
                "error": str(e),
                "timestamp": time.time(),
            }
            await self._emit_event(err_event)
            raise e
