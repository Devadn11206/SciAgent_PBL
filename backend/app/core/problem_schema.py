"""SciAgent Canonical Nonlinear Problem Intermediate Representation (IR).

Built with Pydantic v2. Supports ODE and 1D+time PDE problems, derivative tracking,
parameter specifications, typed initial and boundary conditions, domain normalization,
and factory methods for canonical case studies.
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple, Union
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Core Derivative & Variable Types
# ---------------------------------------------------------------------------

MaxSpatialDerivativeOrder = Literal[0, 1, 2, 3, 4]
MaxTimeDerivativeOrder = Literal[0, 1, 2]
ProblemType = Literal["ODE", "PDE"]
ConservationInvariant = Literal["none", "l2_energy", "mass"]
BCKind = Literal["Dirichlet", "Neumann", "Periodic"]
ConditionRole = Literal["Initial", "Boundary"]


class Domain1D(BaseModel):
    """Specification of an interval domain [min_val, max_val]."""
    min_val: float = Field(..., description="Lower coordinate bound")
    max_val: float = Field(..., description="Upper coordinate bound")

    @model_validator(mode="after")
    def check_bounds(self) -> Domain1D:
        if self.min_val >= self.max_val:
            raise ValueError(f"min_val ({self.min_val}) must be strictly less than max_val ({self.max_val})")
        return self

    @property
    def span(self) -> float:
        return self.max_val - self.min_val


class DomainSpec(BaseModel):
    """Space and/or time domain bounds."""
    t: Domain1D = Field(..., description="Time interval [t_min, t_max]")
    x: Optional[Domain1D] = Field(None, description="Spatial interval [x_min, x_max] for PDEs")

    @property
    def is_spatiotemporal(self) -> bool:
        return self.x is not None


# ---------------------------------------------------------------------------
# Derivative Specification
# ---------------------------------------------------------------------------

class DerivativeSpec(BaseModel):
    """Specifies the highest order derivatives appearing in the differential equation."""
    spatial_order: MaxSpatialDerivativeOrder = Field(
        default=0,
        description="Highest spatial derivative order: up to 4th-order (e.g. u_xxxx)",
    )
    time_order: MaxTimeDerivativeOrder = Field(
        default=1,
        description="Highest time derivative order: up to 2nd-order (e.g. u_tt or x_dd)",
    )


# ---------------------------------------------------------------------------
# Parameter Specification
# ---------------------------------------------------------------------------

class ParameterSpec(BaseModel):
    """Parameter definition for forward simulation or inverse discovery."""
    name: str = Field(..., description="Identifier / symbol key (e.g., 'nu', 'beta')")
    symbol: Optional[str] = Field(None, description="Display glyph or symbol label (e.g., 'ν', 'β')")
    latex_token: Optional[str] = Field(None, description="LaTeX representation (e.g., '\\nu')")
    description: Optional[str] = Field(None, description="Human readable description")
    mode: Literal["fixed", "inverse"] = Field(..., description="'fixed' known parameter or 'inverse' unknown target")
    value: Optional[float] = Field(None, description="True or fixed value if known")
    initial_guess: Optional[float] = Field(None, description="Starting guess for inverse optimization")
    bounds: Optional[Tuple[float, float]] = Field(None, description="Search bounds (lower, upper)")

    @model_validator(mode="after")
    def validate_parameter_consistency(self) -> ParameterSpec:
        if self.mode == "inverse":
            if self.initial_guess is None and self.value is not None:
                self.initial_guess = self.value
            if self.bounds is not None and self.bounds[0] >= self.bounds[1]:
                raise ValueError(f"Parameter '{self.name}' bounds must satisfy bounds[0] < bounds[1]. Got {self.bounds}")
        elif self.mode == "fixed":
            if self.value is None:
                raise ValueError(f"Fixed parameter '{self.name}' must provide a concrete 'value'.")
        return self


# ---------------------------------------------------------------------------
# Conditions (Initial and Boundary)
# ---------------------------------------------------------------------------

class ConditionBase(BaseModel):
    id: str = Field(..., description="Unique condition identifier")
    role: ConditionRole = Field(..., description="'Initial' or 'Boundary'")
    expression: str = Field(..., description="Symbolic/mathematical expression (e.g. 'u(x, 0) = -sin(pi*x)')")
    location: str = Field(..., description="Domain location (e.g. 't = 0', 'x = -1', 'x = +-1')")


class InitialCondition(ConditionBase):
    role: Literal["Initial"] = "Initial"
    order: Literal[0, 1] = Field(
        default=0,
        description="0 for field value condition u(x,0) / x(0), 1 for velocity condition u_t(x,0) / x_dot(0)",
    )
    variable: Optional[str] = Field(None, description="Target variable name (e.g. 'u', 'x', 'psi')")


class BoundaryCondition(ConditionBase):
    role: Literal["Boundary"] = "Boundary"
    kind: BCKind = Field(..., description="Dirichlet, Neumann, or Periodic boundary condition")
    coordinate: Literal["x_min", "x_max", "periodic", "both"] = Field(
        default="both",
        description="Where on the boundary the condition applies",
    )


# ---------------------------------------------------------------------------
# AutoNormalizer
# ---------------------------------------------------------------------------

class DomainTransform(BaseModel):
    """Linear coordinate normalization onto [-1, 1] and chain-rule derivative factors.

    Forward mapping:
        ξ(z) = 2 * (z - z_min) / (z_max - z_min) - 1
             = (2 / L) * z - (z_max + z_min) / L
        where L = z_max - z_min.

    Inverse mapping:
        z(ξ) = z_min + 0.5 * (ξ + 1) * L

    Chain rule for derivatives with respect to physical coordinate z:
        d/dz = (dξ/dz) * d/dξ = (2 / L) * d/dξ
        d^k/dz^k = (2 / L)^k * d^k/dξ^k

    Derivative scaling factor:
        scale_k = (2 / L)^k such that (d^k u / dz^k) = scale_k * (d^k u / dξ^k).
    """
    coord_name: str
    physical_min: float
    physical_max: float
    span: float
    # First derivative chain-rule scale: d_scale = dξ/dz = 2 / (physical_max - physical_min)
    scale_1: float
    scale_2: float
    scale_3: float
    scale_4: float

    @classmethod
    def from_domain(cls, coord_name: str, domain: Domain1D) -> DomainTransform:
        span = domain.span
        scale_1 = 2.0 / span
        return cls(
            coord_name=coord_name,
            physical_min=domain.min_val,
            physical_max=domain.max_val,
            span=span,
            scale_1=scale_1,
            scale_2=scale_1 ** 2,
            scale_3=scale_1 ** 3,
            scale_4=scale_1 ** 4,
        )

    def to_normalized(self, z: float) -> float:
        """Map physical coordinate z in [min, max] to xi in [-1, 1]."""
        return 2.0 * (z - self.physical_min) / self.span - 1.0

    def to_physical(self, xi: float) -> float:
        """Map normalized coordinate xi in [-1, 1] to physical z."""
        return self.physical_min + 0.5 * (xi + 1.0) * self.span

    def get_derivative_scale(self, order: int) -> float:
        """Return the factor (2 / span)^order to convert d^order / dxi^order to d^order / dz^order."""
        if order < 0:
            raise ValueError(f"Derivative order must be non-negative, got {order}")
        if order == 0:
            return 1.0
        return (2.0 / self.span) ** order


class AutoNormalizer(BaseModel):
    """Domain normalizer mapping [x_min, x_max] and [t_min, t_max] to [-1, 1].

    Provides exact chain rule derivative scaling factors:
      dx_scale (k-th order: (2 / (x_max - x_min))^k)
      dt_scale (k-th order: (2 / (t_max - t_min))^k)
    """
    t_transform: DomainTransform
    x_transform: Optional[DomainTransform] = None

    @classmethod
    def from_domain_spec(cls, domain: DomainSpec) -> AutoNormalizer:
        t_tx = DomainTransform.from_domain("t", domain.t)
        x_tx = DomainTransform.from_domain("x", domain.x) if domain.x is not None else None
        return cls(t_transform=t_tx, x_transform=x_tx)

    @property
    def dt_scale(self) -> float:
        """First-order dt scaling factor: d/dt = dt_scale * d/d(t_norm)."""
        return self.t_transform.scale_1

    @property
    def dt2_scale(self) -> float:
        """Second-order dt^2 scaling factor: d^2/dt^2 = dt2_scale * d^2/d(t_norm)^2."""
        return self.t_transform.scale_2

    @property
    def dx_scale(self) -> Optional[float]:
        """First-order dx scaling factor: d/dx = dx_scale * d/d(x_norm)."""
        return self.x_transform.scale_1 if self.x_transform else None

    @property
    def dx2_scale(self) -> Optional[float]:
        """Second-order dx^2 scaling factor: d^2/dx^2 = dx2_scale * d^2/d(x_norm)^2."""
        return self.x_transform.scale_2 if self.x_transform else None

    @property
    def dx3_scale(self) -> Optional[float]:
        """Third-order dx^3 scaling factor."""
        return self.x_transform.scale_3 if self.x_transform else None

    @property
    def dx4_scale(self) -> Optional[float]:
        """Fourth-order dx^4 scaling factor."""
        return self.x_transform.scale_4 if self.x_transform else None

    def get_scaling_factors(self) -> Dict[str, float]:
        """Dictionary of all available derivative chain-rule multipliers."""
        factors: Dict[str, float] = {
            "dt": self.t_transform.scale_1,
            "dtt": self.t_transform.scale_2,
        }
        if self.x_transform is not None:
            factors.update({
                "dx": self.x_transform.scale_1,
                "dxx": self.x_transform.scale_2,
                "dxxx": self.x_transform.scale_3,
                "dxxxx": self.x_transform.scale_4,
            })
        return factors

    def normalize_coords(self, t: float, x: Optional[float] = None) -> Tuple[float, Optional[float]]:
        """Normalize coordinates (t, x) onto [-1, 1]."""
        t_norm = self.t_transform.to_normalized(t)
        x_norm = self.x_transform.to_normalized(x) if (x is not None and self.x_transform) else None
        return t_norm, x_norm

    def denormalize_coords(self, t_norm: float, x_norm: Optional[float] = None) -> Tuple[float, Optional[float]]:
        """De-normalize coordinates from [-1, 1] back to physical domain."""
        t_phys = self.t_transform.to_physical(t_norm)
        x_phys = self.x_transform.to_physical(x_norm) if (x_norm is not None and self.x_transform) else None
        return t_phys, x_phys


# ---------------------------------------------------------------------------
# Canonical Problem Intermediate Representation (IR)
# ---------------------------------------------------------------------------

class ProblemIR(BaseModel):
    """SciAgent Canonical Nonlinear Problem Intermediate Representation (IR)."""
    id: str = Field(..., description="Unique problem slug (e.g. 'burgers', 'duffing')")
    name: str = Field(..., description="Full descriptive title")
    problem_type: ProblemType = Field(..., description="'ODE' or 'PDE'")
    equation_latex: str = Field(..., description="LaTeX representation of the governing equation")
    derivatives: DerivativeSpec = Field(..., description="Order of spatial and time derivatives")
    domain: DomainSpec = Field(..., description="Spatiotemporal domain bounds")
    parameters: List[ParameterSpec] = Field(default_factory=list, description="Parameters in the equation")
    initial_conditions: List[InitialCondition] = Field(default_factory=list, description="Initial conditions")
    boundary_conditions: List[BoundaryCondition] = Field(default_factory=list, description="Boundary conditions (for PDEs)")
    conservation_invariant: ConservationInvariant = Field(
        default="none",
        description="Physical invariant monitored during optimization ('none', 'l2_energy', 'mass')",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context or solver hints")

    @model_validator(mode="after")
    def validate_problem_consistency(self) -> ProblemIR:
        if self.problem_type == "ODE":
            if self.domain.x is not None:
                raise ValueError("ODE problem must not define a spatial domain 'x'.")
            if self.derivatives.spatial_order != 0:
                raise ValueError("ODE problem must have spatial_order = 0.")
            if len(self.boundary_conditions) > 0:
                raise ValueError("ODE problem cannot have boundary conditions; use initial conditions.")
        elif self.problem_type == "PDE":
            if self.domain.x is None:
                raise ValueError("PDE problem must specify a spatial domain 'x'.")
            if self.derivatives.spatial_order == 0:
                raise ValueError("1D+time PDE problem must have spatial_order >= 1.")
        return self

    def get_normalizer(self) -> AutoNormalizer:
        """Create and return an AutoNormalizer configured for this problem's domain."""
        return AutoNormalizer.from_domain_spec(self.domain)

    def get_parameter(self, name: str) -> Optional[ParameterSpec]:
        """Look up a parameter by its name."""
        for p in self.parameters:
            if p.name == name:
                return p
        return None

    @property
    def inverse_parameters(self) -> List[ParameterSpec]:
        """Filter parameters targeted for inverse identification."""
        return [p for p in self.parameters if p.mode == "inverse"]


# ---------------------------------------------------------------------------
# Factory Functions for Canonical Case Studies
# ---------------------------------------------------------------------------

def create_viscous_burgers_ir() -> ProblemIR:
    """Factory for Viscous Burgers' PDE Case Study:

    ∂u/∂t + u * ∂u/∂x = ν * ∂²u/∂x²
    Domain: x ∈ [-1, 1], t ∈ [0, 1]
    IC: u(x, 0) = -sin(πx)
    BC: Dirichlet u(-1, t) = 0, u(1, t) = 0
    Invariant: L2 energy conservation (dissipation tracked)
    Target parameter: ν (inverse identification)
    """
    return ProblemIR(
        id="burgers",
        name="Viscous Burgers' Equation",
        problem_type="PDE",
        equation_latex=r"\frac{\partial u}{\partial t} + u\,\frac{\partial u}{\partial x} = \nu\,\frac{\partial^{2} u}{\partial x^{2}}",
        derivatives=DerivativeSpec(spatial_order=2, time_order=1),
        domain=DomainSpec(
            t=Domain1D(min_val=0.0, max_val=1.0),
            x=Domain1D(min_val=-1.0, max_val=1.0),
        ),
        parameters=[
            ParameterSpec(
                name="nu",
                symbol="ν",
                latex_token=r"\nu",
                description="Kinematic viscosity",
                mode="inverse",
                value=0.01,
                initial_guess=0.02,
                bounds=(0.001, 0.1),
            )
        ],
        initial_conditions=[
            InitialCondition(
                id="burgers_ic",
                expression="u(x, 0) = -sin(pi * x)",
                location="t = 0",
                order=0,
                variable="u",
            )
        ],
        boundary_conditions=[
            BoundaryCondition(
                id="burgers_bc_left",
                kind="Dirichlet",
                expression="u(-1, t) = 0",
                location="x = -1",
                coordinate="x_min",
            ),
            BoundaryCondition(
                id="burgers_bc_right",
                kind="Dirichlet",
                expression="u(+1, t) = 0",
                location="x = +1",
                coordinate="x_max",
            ),
        ],
        conservation_invariant="l2_energy",
        metadata={
            "case_study": "Traffic & Pipeline Shockwaves",
            "reference_solver": "Crank-Nicolson",
            "u_scale": 1.0,
        },
    )


def create_allen_cahn_ir() -> ProblemIR:
    """Factory for Allen-Cahn PDE Case Study:

    ∂u/∂t = ε² * ∂²u/∂x² - (u³ - u)
    Domain: x ∈ [-1, 1], t ∈ [0, 1]
    IC: u(x, 0) = x² * cos(πx)
    BC: Periodic at x = ±1 (u(-1, t) = u(1, t), u_x(-1, t) = u_x(1, t))
    Invariant: none (free-energy gradient flow)
    Target parameter: ε (inverse identification)
    """
    return ProblemIR(
        id="allen-cahn",
        name="Allen-Cahn Equation",
        problem_type="PDE",
        equation_latex=r"\frac{\partial u}{\partial t} = \varepsilon^{2}\,\frac{\partial^{2} u}{\partial x^{2}} - \left(u^{3} - u\right)",
        derivatives=DerivativeSpec(spatial_order=2, time_order=1),
        domain=DomainSpec(
            t=Domain1D(min_val=0.0, max_val=1.0),
            x=Domain1D(min_val=-1.0, max_val=1.0),
        ),
        parameters=[
            ParameterSpec(
                name="eps",
                symbol="ε",
                latex_token=r"\varepsilon",
                description="Interface-width parameter",
                mode="inverse",
                value=0.05,
                initial_guess=0.08,
                bounds=(0.01, 0.2),
            )
        ],
        initial_conditions=[
            InitialCondition(
                id="allen_cahn_ic",
                expression="u(x, 0) = x^2 * cos(pi * x)",
                location="t = 0",
                order=0,
                variable="u",
            )
        ],
        boundary_conditions=[
            BoundaryCondition(
                id="allen_cahn_bc_val",
                kind="Periodic",
                expression="u(-1, t) = u(+1, t)",
                location="x = +-1",
                coordinate="periodic",
            ),
            BoundaryCondition(
                id="allen_cahn_bc_deriv",
                kind="Periodic",
                expression="u_x(-1, t) = u_x(+1, t)",
                location="x = +-1",
                coordinate="periodic",
            ),
        ],
        conservation_invariant="none",
        metadata={
            "case_study": "Li-Ion Battery Phase Separation",
            "reference_solver": "Strang Split-Step FFT",
            "u_scale": 1.0,
        },
    )


def create_cubic_quintic_nlse_ir() -> ProblemIR:
    """Factory for Cubic-Quintic Nonlinear Schrödinger Equation (NLSE) Case Study:

    i * ∂ψ/∂t + (β / 2) * ∂²ψ/∂x² + γ * |ψ|²ψ - δ * |ψ|⁴ψ = 0
    Domain: x ∈ [-6, 6], t ∈ [0, 1] (or z ∈ [0, 1])
    IC: ψ(x, 0) = 1.5 * sech(1.5 * x)
    BC: Periodic and Neumann decay tails at x = ±6
    Invariant: l2_energy (L2 optical power / norm conservation ∫|ψ|² dx)
    Parameters: β (fixed = 1.0), γ (inverse = 1.0), δ (inverse = 0.15)
    """
    return ProblemIR(
        id="nlse",
        name="Cubic-Quintic Nonlinear Schrödinger Equation",
        problem_type="PDE",
        equation_latex=r"i\,\frac{\partial \psi}{\partial t} + \frac{\beta}{2}\,\frac{\partial^{2} \psi}{\partial x^{2}} + \gamma\,|\psi|^{2}\psi - \delta\,|\psi|^{4}\psi = 0",
        derivatives=DerivativeSpec(spatial_order=2, time_order=1),
        domain=DomainSpec(
            t=Domain1D(min_val=0.0, max_val=1.0),
            x=Domain1D(min_val=-6.0, max_val=6.0),
        ),
        parameters=[
            ParameterSpec(
                name="beta",
                symbol="β",
                latex_token=r"\beta",
                description="Dispersion coefficient",
                mode="fixed",
                value=1.0,
            ),
            ParameterSpec(
                name="gamma",
                symbol="γ",
                latex_token=r"\gamma",
                description="Cubic Kerr self-focusing nonlinearity",
                mode="inverse",
                value=1.0,
                initial_guess=0.6,
                bounds=(0.1, 3.0),
            ),
            ParameterSpec(
                name="delta",
                symbol="δ",
                latex_token=r"\delta",
                description="Quintic saturation nonlinearity",
                mode="inverse",
                value=0.15,
                initial_guess=0.05,
                bounds=(0.0, 0.5),
            ),
        ],
        initial_conditions=[
            InitialCondition(
                id="nlse_ic",
                expression="psi(x, 0) = 1.5 * sech(1.5 * x)",
                location="t = 0",
                order=0,
                variable="psi",
            )
        ],
        boundary_conditions=[
            BoundaryCondition(
                id="nlse_bc_periodic",
                kind="Periodic",
                expression="psi(-6, t) = psi(+6, t)",
                location="x = +-6",
                coordinate="periodic",
            ),
            BoundaryCondition(
                id="nlse_bc_neumann",
                kind="Neumann",
                expression="psi_x(+-6, t) = 0",
                location="x = +-6",
                coordinate="both",
            ),
        ],
        conservation_invariant="l2_energy",
        metadata={
            "case_study": "Fiber-Optic Soliton Recovery",
            "reference_solver": "Split-Step FFT",
            "field_channels": ["Re(psi)", "Im(psi)"],
            "u_scale": 1.5,
        },
    )


def create_forced_duffing_ir() -> ProblemIR:
    """Factory for Forced Duffing Oscillator ODE Case Study:

    ẍ + δ * ẋ + α * x + β * x³ = γ * cos(ω * t)
    Domain: t ∈ [0, 40]
    IC: x(0) = 0.5, ẋ(0) = 0.0
    BC: None (ODE)
    Invariant: none
    Parameters:
      delta (fixed = 0.3), alpha (fixed = -1.0), beta (inverse = 1.5, guess = 1.0, bounds=[0.5, 3.0]),
      gamma (fixed = 0.37), omega (fixed = 1.2)
    """
    return ProblemIR(
        id="duffing",
        name="Forced Duffing Oscillator",
        problem_type="ODE",
        equation_latex=r"\ddot{x} + \delta\,\dot{x} + \alpha\,x + \beta\,x^{3} = \gamma\cos(\omega t)",
        derivatives=DerivativeSpec(spatial_order=0, time_order=2),
        domain=DomainSpec(
            t=Domain1D(min_val=0.0, max_val=40.0),
            x=None,
        ),
        parameters=[
            ParameterSpec(
                name="delta",
                symbol="δ",
                latex_token=r"\delta",
                description="Linear damping coefficient",
                mode="fixed",
                value=0.3,
            ),
            ParameterSpec(
                name="alpha",
                symbol="α",
                latex_token=r"\alpha",
                description="Linear stiffness",
                mode="fixed",
                value=-1.0,
            ),
            ParameterSpec(
                name="beta",
                symbol="β",
                latex_token=r"\beta",
                description="Cubic stiffness coefficient",
                mode="inverse",
                value=1.5,
                initial_guess=1.0,
                bounds=(0.5, 3.0),
            ),
            ParameterSpec(
                name="gamma",
                symbol="γ",
                latex_token=r"\gamma",
                description="Periodic forcing amplitude",
                mode="fixed",
                value=0.37,
            ),
            ParameterSpec(
                name="omega",
                symbol="ω",
                latex_token=r"\omega",
                description="Driving angular frequency",
                mode="fixed",
                value=1.2,
            ),
        ],
        initial_conditions=[
            InitialCondition(
                id="duffing_ic_displacement",
                expression="x(0) = 0.5",
                location="t = 0",
                order=0,
                variable="x",
            ),
            InitialCondition(
                id="duffing_ic_velocity",
                expression="x_dot(0) = 0.0",
                location="t = 0",
                order=1,
                variable="x_dot",
            ),
        ],
        boundary_conditions=[],
        conservation_invariant="none",
        metadata={
            "case_study": "Wing Flutter & Grid Chaos",
            "reference_solver": "DOP853",
            "u_scale": 1.2,
        },
    )


CANONICAL_FACTORIES = {
    "burgers": create_viscous_burgers_ir,
    "allen-cahn": create_allen_cahn_ir,
    "nlse": create_cubic_quintic_nlse_ir,
    "duffing": create_forced_duffing_ir,
}


def get_canonical_ir(experiment_id: str) -> ProblemIR:
    """Return pre-built canonical ProblemIR instance by experiment id."""
    if experiment_id not in CANONICAL_FACTORIES:
        raise KeyError(
            f"Unknown experiment ID '{experiment_id}'. Available: {list(CANONICAL_FACTORIES.keys())}"
        )
    return CANONICAL_FACTORIES[experiment_id]()
