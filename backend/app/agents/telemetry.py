"""Telemetry dispatcher for forwarding live epoch metrics and wall-clock timestamps to WebSockets."""

from typing import Callable, Optional, Dict, Any

_EPOCH_DISPATCHER: Optional[Callable[[str, int, Dict[str, Any]], None]] = None

def set_epoch_dispatcher(dispatcher: Callable[[str, int, Dict[str, Any]], None]):
    global _EPOCH_DISPATCHER
    _EPOCH_DISPATCHER = dispatcher

def broadcast_epoch_metric(run_id: str, trial: int, metric: Dict[str, Any]):
    global _EPOCH_DISPATCHER
    if _EPOCH_DISPATCHER is not None:
        try:
            _EPOCH_DISPATCHER(run_id, trial, metric)
        except Exception:
            pass
