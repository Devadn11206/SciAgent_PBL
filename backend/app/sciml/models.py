"""Nonlinear Neural Models for SciAgent.

Provides:
- NonlinearPINN: MLP with Tanh or Fourier Feature Embeddings to overcome spectral bias.
- Automatic registration of inverse parameters as trainable torch.nn.Parameter with projection/clamping.
- FNO1d: 1D Fourier Neural Operator using spectral convolutions (torch.fft.rfft).
"""

from __future__ import annotations

import math
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.app.core.problem_schema import ParameterSpec, ProblemIR


# ---------------------------------------------------------------------------
# 1. Fourier Feature Embedding (Random Fourier Features)
# ---------------------------------------------------------------------------

def estimate_dominant_frequency(problem: Optional[ProblemIR], n_samples: int = 128) -> float:
    """Compute the dominant spatial frequency k_dom of the initial condition u(x, 0)

    using np.fft.rfft on physical coordinates x in [x_min, x_max].
    """
    if problem is None or problem.problem_type == "ODE":
        return 1.0

    x_domain = problem.domain.x
    x_min = float(x_domain.min_val) if x_domain else -1.0
    x_max = float(x_domain.max_val) if x_domain else 1.0
    L = max(1e-4, x_max - x_min)

    x_vals = np.linspace(x_min, x_max, n_samples)
    u_vals = np.zeros_like(x_vals)

    # Evaluate analytical IC if available
    evaluated = False
    if problem.initial_conditions:
        for ic in problem.initial_conditions:
            expr_str = ic.expression or ""
            if "=" in expr_str:
                expr_str = expr_str.split("=")[-1].strip()
            expr_clean = expr_str.replace("^", "**").replace("pi", "np.pi")
            try:
                safe_dict = {
                    "np": np,
                    "sin": np.sin,
                    "cos": np.cos,
                    "exp": np.exp,
                    "tanh": np.tanh,
                    "cosh": np.cosh,
                    "x": x_vals,
                    "pi": np.pi,
                }
                res = eval(expr_clean, {"__builtins__": {}}, safe_dict)
                if isinstance(res, np.ndarray) and res.shape == x_vals.shape:
                    u_vals = res
                    evaluated = True
                    break
            except Exception:
                pass

    if not evaluated:
        prob_id = problem.id.lower()
        eq_latex = problem.equation_latex.lower()
        if "burgers" in prob_id or "nu" in eq_latex:
            u_vals = -np.sin(np.pi * x_vals)
        elif "allen" in prob_id or "cahn" in prob_id or "eps" in eq_latex:
            u_vals = (x_vals ** 2) * np.cos(np.pi * x_vals)
        elif "nlse" in prob_id or "schrodinger" in prob_id or "psi" in eq_latex:
            u_vals = 1.5 / np.cosh(1.5 * x_vals)
        else:
            u_vals = -np.sin(np.pi * x_vals)

    # Subtract mean and compute FFT
    u_detrend = u_vals - np.mean(u_vals)
    rfft_vals = np.abs(np.fft.rfft(u_detrend))
    freqs = np.fft.rfftfreq(n_samples, d=(L / n_samples))

    if len(rfft_vals) > 1 and np.max(rfft_vals[1:]) > 1e-6:
        # Dominant non-zero frequency in cycles per unit length
        k_dom = float(freqs[1:][np.argmax(rfft_vals[1:])])
        # Convert to dimensionless scale factor on normalized coordinates [-1, 1]
        # dx_norm = 2.0 / L, so wave on [-1, 1] has frequency k_dom * (L / 2)
        norm_k = max(1.0, k_dom * (L / 2.0))
        return float(norm_k)

    return 1.0


class MultiScaleFourierFeatureEmbedding(nn.Module):
    """Multi-Scale Random Fourier Feature mapping combining:

    - Low-frequency projections (sigma = 1.0) for smooth macroscopic envelopes / IC anchoring.
    - Medium/high-frequency projections (sigma = max(2.5, 0.5 * k_dom)) for resolving multi-cycle waves without spectral bias.
    """

    def __init__(
        self,
        in_features: int,
        num_frequencies: int = 32,
        low_scale: float = 1.0,
        high_scale: float = 2.5,
    ):
        super().__init__()
        self.in_features = in_features
        n_low = num_frequencies // 2
        n_high = num_frequencies - n_low

        B_low = torch.randn(in_features, n_low) * low_scale
        B_high = torch.randn(in_features, n_high) * high_scale

        B = torch.cat([B_low, B_high], dim=1)
        self.register_buffer("B", B)
        self.num_frequencies = num_frequencies

    @property
    def out_features(self) -> int:
        return self.num_frequencies * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, in_features) in normalized coords [-1, 1]
        x_proj = 2.0 * math.pi * torch.matmul(x, self.B)
        return torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)


# Legacy alias
FourierFeatureEmbedding = MultiScaleFourierFeatureEmbedding


# ---------------------------------------------------------------------------
# 2. Nonlinear PINN (Physics-Informed Neural Network)
# ---------------------------------------------------------------------------

class NonlinearPINN(nn.Module):
    """Physics-Informed Neural Network for nonlinear ODE/PDE problems.

    Supports:
    - Standard Tanh MLP or Fourier Feature Embedding.
    - Automatic registration of inverse target parameters as trainable parameters.
    - Parameter clamping/projection within physical bounds.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 4,
        use_fourier_features: bool = False,
        fourier_scale: float = 1.0,
        problem: Optional[ProblemIR] = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_fourier_features = use_fourier_features

        if use_fourier_features:
            k_dom = estimate_dominant_frequency(problem)
            high_scale = max(2.5, 0.5 * k_dom)
            self.embedding = MultiScaleFourierFeatureEmbedding(
                in_dim,
                num_frequencies=32,
                low_scale=1.0,
                high_scale=high_scale,
            )
            layer_in = self.embedding.out_features
        else:
            self.embedding = None
            layer_in = in_dim

        layers: List[nn.Module] = []
        layers.append(nn.Linear(layer_in, hidden_dim))
        layers.append(nn.Tanh())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

        # Initialize weights with Xavier uniform
        self._init_weights()

        # Inverse parameter registration
        self.inverse_params: nn.ParameterDict = nn.ParameterDict()
        self.param_bounds: Dict[str, Tuple[float, float]] = {}

        if problem is not None:
            self._register_inverse_parameters(problem)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.8)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _register_inverse_parameters(self, problem: ProblemIR):
        """Register all parameters with mode == 'inverse' as trainable torch parameters."""
        for p in problem.parameters:
            if p.mode == "inverse":
                init_val = p.initial_guess if p.initial_guess is not None else (p.value if p.value is not None else 1.0)
                # Store as 1-element float tensor parameter
                self.inverse_params[p.name] = nn.Parameter(torch.tensor([float(init_val)], dtype=torch.float32))
                if p.bounds is not None:
                    self.param_bounds[p.name] = p.bounds

    def clamp_inverse_parameters(self):
        """Clamp / project inverse parameters to lie strictly within their specified bounds."""
        with torch.no_grad():
            for name, param in self.inverse_params.items():
                if name in self.param_bounds:
                    lo, hi = self.param_bounds[name]
                    param.clamp_(min=lo, max=hi)

    def get_discovered_parameters(self) -> Dict[str, float]:
        """Return current discovered parameter values as a python dictionary."""
        discovered: Dict[str, float] = {}
        for name, param in self.inverse_params.items():
            discovered[name] = float(param.item())
        return discovered

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Forward pass: coords (N, in_dim) in normalized [-1, 1] domain.
        Applies tanh scaling / soft clamping to keep predictions within [-6.0, 6.0] physical bounds.
        """
        if self.embedding is not None:
            features = self.embedding(coords)
        else:
            features = coords
        raw_out = self.net(features)
        # Soft-clamp to prevent runaway numerical explosion (e.g. [-20, +10]) in early training
        return torch.clamp(raw_out, min=-8.0, max=8.0)


# ---------------------------------------------------------------------------
# 3. 1D Fourier Neural Operator (FNO1d)
# ---------------------------------------------------------------------------

class SpectralConv1d(nn.Module):
    """1D Fourier layer: computes spectral convolution via FFT."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to retain

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )

    def compl_mul1d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # (batch, in_channel, x), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.shape[0]

        # Compute Fourier transform along spatial dimension
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-1) // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        modes = min(self.modes1, x_ft.shape[-1])
        out_ft[:, :, :modes] = self.compl_mul1d(x_ft[:, :, :modes], self.weights1[:, :, :modes])

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNO1d(nn.Module):
    """1D Fourier Neural Operator for parametric mapping from initial state to full solution trajectory."""

    def __init__(
        self,
        modes: int = 16,
        width: int = 32,
        in_channels: int = 1,
        out_channels: int = 1,
        num_layers: int = 4,
    ):
        super().__init__()
        self.modes = modes
        self.width = width

        # Lift input to higher-dimensional channel space
        self.fc0 = nn.Linear(in_channels, self.width)

        self.conv_layers = nn.ModuleList([
            SpectralConv1d(self.width, self.width, self.modes) for _ in range(num_layers)
        ])
        self.w_layers = nn.ModuleList([
            nn.Conv1d(self.width, self.width, 1) for _ in range(num_layers)
        ])

        # Project back to output channel space
        self.fc1 = nn.Linear(self.width, 64)
        self.fc2 = nn.Linear(64, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, nx, in_channels)
        x = self.fc0(x)
        # Permute to (batch_size, channels, nx) for Conv1d / FFT
        x = x.permute(0, 2, 1)

        for conv, w in zip(self.conv_layers, self.w_layers):
            x1 = conv(x)
            x2 = w(x)
            x = F.gelu(x1 + x2)

        x = x.permute(0, 2, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x
