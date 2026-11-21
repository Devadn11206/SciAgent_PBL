"""Ollama & LLM Client with Strict Pydantic v2 Output Schemas and Offline Fallback.

Ensures:
- Direct local Ollama communication (qwen2.5:7b-instruct, qwen2.5:3b-instruct, llama3.1:8b).
- Automatic deterministic offline rule-based fallback if Ollama service is unreachable.
- Typed output schemas:
    * ProblemDiagnosis
    * SolverPlan
    * CriticVerdict
    * ResearchSummary
- LLMs ONLY output typed diagnoses, configurations, and insights (NEVER compute raw numbers).
"""

from __future__ import annotations

import json
import time
import urllib.request
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, TypeVar
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Strict Pydantic Output Schemas
# ---------------------------------------------------------------------------

class ProblemDiagnosis(BaseModel):
    problem_type: Literal["ODE", "PDE"]
    classified_regime: str = Field(..., description="e.g. shock-forming, stiff phase-separation, dispersive soliton, chaotic ODE")
    is_nonlinear: bool
    mode: Literal["forward", "inverse"]
    critical_features: List[str] = Field(..., description="Identified physical dynamics (shocks, boundary layers, etc.)")
    estimated_time_scale: float = Field(..., description="Estimated CFL or characteristic timescale")
    recommended_sampling: str = Field(..., description="Sampling recommendation (e.g. adaptive, uniform)")


class SolverPlan(BaseModel):
    numerical_reference_method: str = Field(..., description="Crank-Nicolson, Strang Split-Step FFT, or DOP853")
    pinn_hidden_dim: int = Field(64, description="MLP hidden layer dimension")
    pinn_num_layers: int = Field(4, description="Number of hidden layers")
    use_fourier_features: bool = Field(False, description="Enable random Fourier feature embeddings")
    use_causal_weighting: bool = Field(False, description="Enable causal temporal loss weights")
    use_rar: bool = Field(False, description="Enable residual-adaptive refinement")
    adam_epochs: int = Field(250, description="Adam optimizer iterations")
    lbfgs_epochs: int = Field(20, description="L-BFGS fine-tuning iterations")
    reasoning: str = Field(..., description="Method selection rationale")


class CriticVerdict(BaseModel):
    verdict: Literal["ACCEPT", "SELF_CORRECT", "FAIL"]
    current_trial: int
    rel_l2_error: float
    conservation_drift_pct: float
    tolerances_exceeded: List[str]
    root_cause_diagnosis: str = Field(..., description="Physical or optimization diagnosis of the bottleneck")
    corrective_actions: List[str] = Field(..., description="Actionable interventions for next trial (e.g. ['RAR +2000 pts', 'Causal weighting'])")
    updated_plan: Optional[SolverPlan] = None


class ResearchSummary(BaseModel):
    title: str
    headline: str
    key_findings: List[str]
    physical_interpretation: List[str]
    ablation_takeaways: List[str]
    confidence_level: Literal["High", "Moderate", "Preliminary"]


# ---------------------------------------------------------------------------
# Ollama Client with Offline Fallback
# ---------------------------------------------------------------------------

T = TypeVar("T", bound=BaseModel)


class OllamaClient:
    """Lightweight HTTP client for local Ollama instances with offline rule-based fallback."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        preferred_models: Optional[List[str]] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.preferred_models = preferred_models or [
            "qwen2.5:3b-instruct",
            "qwen3.5:2b",
            "llama3.2:3b",
            "qwen2.5:7b-instruct",
            "llama3.1:8b",
        ]
        self._last_health_check = 0.0
        self._cached_online = False
        self._cached_model = None

    def check_health(self) -> Tuple[bool, Optional[str]]:
        """Check if Ollama server is running and detect available models (cached for 15s)."""
        now = time.time()
        if now - self._last_health_check < 15.0:
            return self._cached_online, self._cached_model

        self._last_health_check = now
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", headers={"User-Agent": "SciAgent/1.0"})
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode())
                    available = [m["name"] for m in data.get("models", [])]
                    for pm in self.preferred_models:
                        if pm in available or any(pm.split(":")[0] in av for av in available):
                            self._active_model = pm
                            self._is_online = True
                            self._cached_online = True
                            self._cached_model = pm
                            return True, pm
                    # Default to first available
                    if available:
                        self._active_model = available[0]
                        self._is_online = True
                        self._cached_online = True
                        self._cached_model = available[0]
                        return True, available[0]
        except Exception:
            pass
        self._is_online = False
        self._cached_online = False
        self._cached_model = None
        return False, None

    @property
    def is_online(self) -> bool:
        if self._is_online is None:
            online, _ = self.check_health()
            return online
        return self._is_online

    def generate_structured(
        self,
        prompt: str,
        system_prompt: str,
        schema: Type[T],
        fallback_factory: Any,
    ) -> T:
        """Query Ollama with JSON mode constraint and parse into Pydantic schema, falling back to deterministic rules if offline."""
        online, model = self.check_health()
        if not online or not model:
            return fallback_factory()

        full_prompt = (
            f"{system_prompt}\n\n"
            f"Respond ONLY with a valid JSON object matching this schema:\n"
            f"{json.dumps(schema.model_json_schema())}\n\n"
            f"Context / Query:\n{prompt}\n\n"
            f"JSON Output:"
        )

        payload = {
            "model": model,
            "prompt": full_prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1, "num_predict": 700},
        }

        try:
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{self.base_url}/api/generate",
                data=req_data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=18.0) as resp:
                res_json = json.loads(resp.read().decode())
                response_text = res_json.get("response", "{}")
                parsed_dict = json.loads(response_text)
                return schema.model_validate(parsed_dict)
        except Exception:
            # Automatic graceful fallback to deterministic engine
            return fallback_factory()
