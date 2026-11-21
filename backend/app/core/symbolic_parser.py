"""SciAgent Symbolic Parser using SymPy.

Features:
1. Parses mathematical strings/LaTeX into SymPy expressions, cleanly separating
   dependent variables (u, psi, x for ODE), independent variables (x, t, z),
   and physical/symbolic parameters (nu, eps, gamma, delta, alpha, beta, omega).
2. Detects nonlinear terms (e.g. u*u_x, u^3, |psi|^2*psi) and classifies
   the PDE/ODE regime (parabolic, hyperbolic, dispersive, stiff).
3. Pre-flight well-posedness & stability audits:
   - Verifies condition counts match highest spatial & temporal orders.
   - Computes CFL stability bounds for numerical discretizations.
4. Generates Typst math syntax via `to_typst_math()` for single-column reports.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from backend.app.core.problem_schema import (
    BoundaryCondition,
    InitialCondition,
    ProblemIR,
)


# ---------------------------------------------------------------------------
# Reserved Names and Dictionaries
# ---------------------------------------------------------------------------

DEFAULT_INDEPENDENT_VARS = {"x", "t", "z", "y"}
DEFAULT_DEPENDENT_VARS = {"u", "psi", "v", "w", "phi"}
# Note: In ODEs, 'x' might be the dependent variable if 't' is time.
DEFAULT_PARAMETERS = {
    "nu",
    "eps",
    "epsilon",
    "gamma",
    "delta",
    "alpha",
    "beta",
    "omega",
    "k",
    "c",
    "mu",
    "lambda_",
    "sigma",
    "rho",
}

TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


# ---------------------------------------------------------------------------
# Data Classes for Analysis Output
# ---------------------------------------------------------------------------

RegimeType = Literal["parabolic", "hyperbolic", "dispersive", "stiff", "elliptic", "undetermined"]


@dataclass
class NonlinearTermInfo:
    term_expr: str
    description: str
    kind: str  # "advective", "reaction/polynomial", "cubic_quintic", "generic"


@dataclass
class ClassificationResult:
    problem_type: Literal["ODE", "PDE"]
    spatial_order: int
    time_order: int
    is_nonlinear: bool
    regimes: List[RegimeType]
    primary_regime: RegimeType
    nonlinear_terms: List[NonlinearTermInfo]
    diagnostics: List[str]


@dataclass
class CFLStabilityBound:
    cfl_type: Literal["advective", "diffusive", "dispersive", "reaction"]
    condition_str: str
    max_dt_formula: str
    numerical_max_dt: Optional[float] = None
    notes: Optional[str] = None


@dataclass
class WellPosednessAudit:
    is_well_posed: bool
    spatial_order: int
    time_order: int
    num_boundary_conditions: int
    num_initial_conditions: int
    required_bc_count: int
    required_ic_count: int
    status: Literal["PASSED", "WARNING", "FAILED"]
    messages: List[str]
    cfl_bounds: List[CFLStabilityBound] = field(default_factory=list)


# ---------------------------------------------------------------------------
# LaTeX Pre-processing and Parsing
# ---------------------------------------------------------------------------

def normalize_equation_str(eq_str: str) -> str:
    """Clean and normalize equation LaTeX / string into an equation expression."""
    s = eq_str.strip()
    # Strip math delimiters
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    if s.startswith(r"\[") and s.endswith(r"\]"):
        s = s[2:-2].strip()

    # Normalize unicode / greek
    s = s.replace(r"\nu", "nu").replace("ν", "nu")
    s = s.replace(r"\varepsilon", "eps").replace(r"\eps", "eps").replace("ε", "eps")
    s = s.replace(r"\gamma", "gamma").replace("γ", "gamma")
    s = s.replace(r"\delta", "delta").replace("δ", "delta")
    s = s.replace(r"\alpha", "alpha").replace("α", "alpha")
    s = s.replace(r"\beta", "beta").replace("β", "beta")
    s = s.replace(r"\omega", "omega").replace("ω", "omega")
    s = s.replace(r"\psi", "psi").replace("ψ", "psi")
    s = s.replace(r"\pi", "pi").replace("π", "pi")

    # Handle derivatives written in LaTeX with or without braces in powers:
    # \frac{\partial^{2} u}{\partial x^{2}} or \frac{\partial^2 u}{\partial x^2}
    s = re.sub(
        r"\\frac\{\\partial(?:\^\{?4\}?)\s*([a-zA-Z]+)\}\{\\partial\s*([a-zA-Z])(?:\^\{?4\}?)\}",
        r"Derivative(\1, \2, \2, \2, \2)",
        s,
    )
    s = re.sub(
        r"\\frac\{\\partial(?:\^\{?3\}?)\s*([a-zA-Z]+)\}\{\\partial\s*([a-zA-Z])(?:\^\{?3\}?)\}",
        r"Derivative(\1, \2, \2, \2)",
        s,
    )
    s = re.sub(
        r"\\frac\{\\partial(?:\^\{?2\}?)\s*([a-zA-Z]+)\}\{\\partial\s*([a-zA-Z])(?:\^\{?2\}?)\}",
        r"Derivative(\1, \2, \2)",
        s,
    )
    s = re.sub(
        r"\\frac\{\\partial\s*([a-zA-Z]+)\}\{\\partial\s*([a-zA-Z])\}",
        r"Derivative(\1, \2)",
        s,
    )

    # Ordinary derivatives: \frac{d^2 x}{d t^2} or \frac{d x}{d t}
    s = re.sub(
        r"\\frac\{d(?:\^\{?2\}?)\s*([a-zA-Z]+)\}\{d\s*([a-zA-Z])(?:\^\{?2\}?)\}",
        r"Derivative(\1, \2, \2)",
        s,
    )
    s = re.sub(
        r"\\frac\{d\s*([a-zA-Z]+)\}\{d\s*([a-zA-Z])\}",
        r"Derivative(\1, \2)",
        s,
    )

    # ODE dots: \ddot{x} -> Derivative(x, t, t), \dot{x} -> Derivative(x, t)
    s = re.sub(r"\\ddot\{([a-zA-Z]+)\}", r"Derivative(\1, t, t)", s)
    s = re.sub(r"\\dot\{([a-zA-Z]+)\}", r"Derivative(\1, t)", s)
    s = s.replace("ẍ", "Derivative(x, t, t)").replace("ẋ", "Derivative(x, t)")

    # Modulus notation: |\psi|^2 \psi or |\psi|^4 \psi
    s = re.sub(r"\|([a-zA-Z]+)\|\^\{?4\}?", r"(Abs(\1)**4)", s)
    s = re.sub(r"\|([a-zA-Z]+)\|\^\{?2\}?", r"(Abs(\1)**2)", s)

    # Trig functions and explicit multiplication with symbols (e.g. gamma\cos(omega t) -> gamma * cos(omega * t))
    s = re.sub(r"([a-zA-Z0-9_]+)\s*\\cos", r"\1 * cos", s)
    s = re.sub(r"([a-zA-Z0-9_]+)\s*\\sin", r"\1 * sin", s)
    s = re.sub(r"([a-zA-Z0-9_]+)\s*\\exp", r"\1 * exp", s)
    s = s.replace(r"\cos", "cos").replace(r"\sin", "sin").replace(r"\exp", "exp")

    # Fractions like \frac{\beta}{2} -> (\beta / 2)
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", s)

    # Braces in simple exponents like x^{3} -> x**3
    s = re.sub(r"\^\{([^{}]+)\}", r"**(\1)", s)
    s = s.replace("^", "**")

    # Remove LaTeX spacing commands and markers
    s = s.replace(r"\,", " ").replace(r"\;", " ").replace(r"\quad", " ")
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace("{", "(").replace("}", ")")

    return s


def parse_symbolic_equation(
    equation_str: str,
    independent_vars: Optional[Set[str]] = None,
    dependent_vars: Optional[Set[str]] = None,
    parameter_names: Optional[Set[str]] = None,
) -> Tuple[sp.Expr, Dict[str, sp.Symbol], Dict[str, Any]]:
    """Parse equation string or LaTeX into a SymPy expression (equated to zero: lhs - rhs = 0).

    Returns:
        (residual_expr, symbols_map, classification_context)
    """
    clean_str = normalize_equation_str(equation_str)

    # Split equality
    if "=" in clean_str:
        parts = clean_str.split("=")
        lhs_str, rhs_str = parts[0].strip(), parts[1].strip()
    else:
        lhs_str, rhs_str = clean_str, "0"

    # Define symbol environment
    indep = set(independent_vars or DEFAULT_INDEPENDENT_VARS)
    dep = set(dependent_vars or DEFAULT_DEPENDENT_VARS)
    params = set(parameter_names or DEFAULT_PARAMETERS)

    # Local symbols dictionary for SymPy parser
    local_dict: Dict[str, Any] = {
        "Derivative": sp.Derivative,
        "Abs": sp.Abs,
        "cos": sp.cos,
        "sin": sp.sin,
        "exp": sp.exp,
        "pi": sp.pi,
        "I": sp.I,
        "i": sp.I,
    }

    all_symbols = indep | dep | params | {"t", "x", "z", "u", "psi", "nu", "eps", "gamma", "delta", "alpha", "beta", "omega"}
    for name in all_symbols:
        local_dict[name] = sp.Symbol(name)

    lhs_expr = parse_expr(lhs_str, local_dict=local_dict, transformations=TRANSFORMATIONS)
    rhs_expr = parse_expr(rhs_str, local_dict=local_dict, transformations=TRANSFORMATIONS)
    # Do not call sp.simplify() because in SymPy Derivative(Symbol, Symbol) simplifies to 0!
    # Instead, use basic collection or subtraction directly.
    residual_expr = lhs_expr - rhs_expr

    # Discovered symbols in the expression
    # Note: atoms(Derivative) contains wrt symbols which free_symbols might omit when gen is a Symbol
    all_free_syms = set(residual_expr.free_symbols)
    derivatives = list(residual_expr.atoms(sp.Derivative))
    for d in derivatives:
        for arg in d.args:
            if isinstance(arg, tuple):
                for sub in arg:
                    if isinstance(sub, sp.Symbol):
                        all_free_syms.add(sub)
            elif isinstance(arg, sp.Symbol):
                all_free_syms.add(arg)

    free_syms = {s.name: s for s in all_free_syms if isinstance(s, sp.Symbol)}
    parsed_indep = {k: v for k, v in free_syms.items() if k in indep}
    parsed_dep = {k: v for k, v in free_syms.items() if k in dep}
    parsed_params = {k: v for k, v in free_syms.items() if k in params or (k not in indep and k not in dep and k not in ("I", "i"))}

    # Detect if 'x' was treated as an ODE dependent variable (e.g. x_dot + x = 0 with independent t)
    derivatives = list(residual_expr.atoms(sp.Derivative))
    for d in derivatives:
        deriv_target = str(d.args[0])
        deriv_wrt = [str(arg) if isinstance(arg, sp.Symbol) else str(arg[0]) for arg in d.args[1:]]
        if "t" in deriv_wrt and deriv_target == "x":
            # x is dependent variable in ODE
            if "x" in parsed_indep:
                del parsed_indep["x"]
            parsed_dep["x"] = free_syms.get("x", sp.Symbol("x"))

    context = {
        "independent_vars": parsed_indep,
        "dependent_vars": parsed_dep,
        "parameters": parsed_params,
        "derivatives": derivatives,
    }

    return residual_expr, free_syms, context


# ---------------------------------------------------------------------------
# Nonlinear Term Detection and PDE/ODE Classification
# ---------------------------------------------------------------------------

class SymbolicClassifier:
    """Detects nonlinearities and determines mathematical regime of ODE/PDE."""

    @classmethod
    def analyze(
        cls,
        residual_expr: sp.Expr,
        problem_type: Optional[Literal["ODE", "PDE"]] = None,
        dependent_vars: Optional[List[str]] = None,
    ) -> ClassificationResult:
        derivatives = list(residual_expr.atoms(sp.Derivative))

        # Detect spatial and time orders
        spatial_order = 0
        time_order = 0

        for d in derivatives:
            # d.args = (expr, (var, order), ...)
            wrt_symbols = []
            for arg in d.args[1:]:
                if isinstance(arg, (tuple, sp.Tuple)):
                    sym = str(arg[0])
                    order_val = int(arg[1]) if len(arg) > 1 else 1
                    wrt_symbols.extend([sym] * order_val)
                elif isinstance(arg, sp.Symbol):
                    wrt_symbols.append(str(arg))
                else:
                    try:
                        sym = str(arg[0])
                        order_val = int(arg[1]) if len(arg) > 1 else 1
                        wrt_symbols.extend([sym] * order_val)
                    except Exception:
                        wrt_symbols.append(str(arg))

            t_count = sum(1 for v in wrt_symbols if v == "t")
            x_count = sum(1 for v in wrt_symbols if v in ("x", "z", "y"))

            if t_count > time_order:
                time_order = t_count
            if x_count > spatial_order:
                spatial_order = x_count

        if problem_type is None:
            problem_type = "ODE" if spatial_order == 0 else "PDE"

        # Expand residual to inspect additive terms
        expanded = sp.expand(residual_expr)
        terms = sp.Add.make_args(expanded)

        nonlinear_terms: List[NonlinearTermInfo] = []
        is_nonlinear = False
        regimes: List[RegimeType] = []
        diagnostics: List[str] = []

        dep_names = set(dependent_vars or ["u", "psi", "x"])

        for term in terms:
            term_str = str(term)

            # 1. Advective nonlinearity: u * u_x or similar
            has_dep = any(var in term_str for var in dep_names)
            term_derivatives = list(term.atoms(sp.Derivative))

            if term_derivatives and has_dep:
                # Check if multiplied by dependent variable
                for d in term_derivatives:
                    quotient = term / d
                    if any(sp.Symbol(var) in quotient.free_symbols for var in dep_names):
                        is_nonlinear = True
                        nonlinear_terms.append(
                            NonlinearTermInfo(
                                term_expr=term_str,
                                description="Convective / advective nonlinear flux (u * ∂u/∂x)",
                                kind="advective",
                            )
                        )
                        if "hyperbolic" not in regimes:
                            regimes.append("hyperbolic")

            # 2. Polynomial reaction: u^3, u^5, etc.
            # Avoid sp.degree() directly on non-polynomial expressions containing Abs
            if not term.atoms(sp.Abs):
                for var in dep_names:
                    var_sym = sp.Symbol(var)
                    if var_sym in term.free_symbols:
                        try:
                            deg = sp.degree(term, gen=var_sym)
                            if deg and deg > 1:
                                is_nonlinear = True
                                nonlinear_terms.append(
                                    NonlinearTermInfo(
                                        term_expr=term_str,
                                        description=f"Polynomial reaction/stiffness of order {deg} ({var}^{deg})",
                                        kind="reaction/polynomial",
                                    )
                                )
                                if deg >= 3 and "stiff" not in regimes:
                                    regimes.append("stiff")
                        except Exception:
                            pass

            # 3. Soliton / NLSE complex nonlinearities (|psi|^2 * psi, |psi|^4 * psi)
            if "Abs(" in term_str:
                is_nonlinear = True
                nonlinear_terms.append(
                    NonlinearTermInfo(
                        term_expr=term_str,
                        description="Kerr / quintic saturation optical nonlinearity (|psi|^n * psi)",
                        kind="cubic_quintic",
                    )
                )
                if "dispersive" not in regimes:
                    regimes.append("dispersive")

        # Check for linear dispersion or diffusion
        has_diffusion = False
        has_imaginary_time = (sp.I in residual_expr.atoms() or sp.Symbol("I") in residual_expr.free_symbols)

        for d in derivatives:
            wrt = [str(a[0]) if isinstance(a, tuple) else str(a) for a in d.args[1:]]
            if wrt.count("x") == 2:
                has_diffusion = True

        if has_imaginary_time:
            if "dispersive" not in regimes:
                regimes.append("dispersive")
        elif has_diffusion and time_order == 1:
            if "parabolic" not in regimes:
                regimes.append("parabolic")

        if not regimes:
            regimes.append("hyperbolic" if spatial_order == 1 else "undetermined")

        primary_regime = regimes[0]

        diagnostics.append(
            f"Classification: {problem_type} with spatial order {spatial_order}, time order {time_order}."
        )
        if is_nonlinear:
            diagnostics.append(f"Detected {len(nonlinear_terms)} nonlinear term(s). Regime: {primary_regime}.")
        else:
            diagnostics.append("Linear differential operator.")

        return ClassificationResult(
            problem_type=problem_type,
            spatial_order=spatial_order,
            time_order=time_order,
            is_nonlinear=is_nonlinear,
            regimes=regimes,
            primary_regime=primary_regime,
            nonlinear_terms=nonlinear_terms,
            diagnostics=diagnostics,
        )


# ---------------------------------------------------------------------------
# Pre-Flight Well-Posedness and Stability Checks
# ---------------------------------------------------------------------------

class PreFlightAuditor:
    """Audits well-posedness (IC/BC counts) and calculates CFL stability limits."""

    @classmethod
    def audit(
        cls,
        problem: ProblemIR,
        dx_val: Optional[float] = None,
        max_u_val: Optional[float] = None,
    ) -> WellPosednessAudit:
        spatial_order = problem.derivatives.spatial_order
        time_order = problem.derivatives.time_order

        # Count conditions
        ic_count = len(problem.initial_conditions)
        bc_count = len(problem.boundary_conditions)

        # Standard requirements:
        # Time order N requires N initial conditions (e.g. u(0) for order 1, u(0) and u_dot(0) for order 2)
        required_ic = time_order
        # Spatial order M in 1D requires M boundary conditions (or 1 periodic pair counting as 2)
        # Note: Periodic BC counts both boundaries
        effective_bc_count = 0
        for bc in problem.boundary_conditions:
            if bc.kind == "Periodic":
                effective_bc_count += 2
            elif bc.coordinate in ("both", "periodic"):
                effective_bc_count += 2
            else:
                effective_bc_count += 1

        required_bc = spatial_order if problem.problem_type == "PDE" else 0

        messages: List[str] = []
        is_well_posed = True

        # Check ICs
        if ic_count < required_ic:
            is_well_posed = False
            messages.append(
                f"Under-specified initial conditions: provided {ic_count}, required {required_ic} for order-{time_order} time derivative."
            )
        elif ic_count > required_ic:
            messages.append(
                f"Notice: provided {ic_count} ICs for order-{time_order} time derivative (over-specified or auxiliary)."
            )
        else:
            messages.append(f"Initial conditions check passed: {ic_count}/{required_ic} conditions supplied.")

        # Check BCs for PDEs
        if problem.problem_type == "PDE":
            if effective_bc_count < required_bc:
                is_well_posed = False
                messages.append(
                    f"Under-specified boundary conditions: effective {effective_bc_count}, required {required_bc} for order-{spatial_order} spatial derivative."
                )
            elif effective_bc_count > required_bc:
                messages.append(
                    f"Boundary conditions: effective {effective_bc_count} provided for order-{spatial_order} spatial operator."
                )
            else:
                messages.append(f"Boundary conditions check passed: {effective_bc_count}/{required_bc} boundary values supplied.")
        else:
            if bc_count > 0:
                messages.append(f"Warning: ODE has {bc_count} boundary conditions; typically ODEs specify initial values.")

        # Compute CFL Bounds
        cfl_bounds = cls.compute_cfl_bounds(problem, dx_val=dx_val, max_u_val=max_u_val)

        status: Literal["PASSED", "WARNING", "FAILED"] = (
            "PASSED" if is_well_posed else "FAILED"
        )

        return WellPosednessAudit(
            is_well_posed=is_well_posed,
            spatial_order=spatial_order,
            time_order=time_order,
            num_boundary_conditions=bc_count,
            num_initial_conditions=ic_count,
            required_bc_count=required_bc,
            required_ic_count=required_ic,
            status=status,
            messages=messages,
            cfl_bounds=cfl_bounds,
        )

    @classmethod
    def compute_cfl_bounds(
        cls,
        problem: ProblemIR,
        dx_val: Optional[float] = None,
        max_u_val: Optional[float] = None,
    ) -> List[CFLStabilityBound]:
        bounds: List[CFLStabilityBound] = []

        # Default physical grid spacing assumption if none provided
        if dx_val is None:
            if problem.domain.x is not None:
                dx_val = problem.domain.x.span / 128.0
            else:
                dx_val = 0.01

        u_scale = max_u_val or float(problem.metadata.get("u_scale", 1.0))

        # Check for specific canonical experiments or general terms
        if problem.id == "burgers":
            # Convective CFL: dt <= dx / max|u|
            dt_advective = dx_val / u_scale
            bounds.append(
                CFLStabilityBound(
                    cfl_type="advective",
                    condition_str="Δt ≤ Δx / max|u|",
                    max_dt_formula="dx / max|u|",
                    numerical_max_dt=dt_advective,
                    notes=f"Advective CFL: Δx={dx_val:.4f}, max|u|={u_scale:.2f} -> Δt ≤ {dt_advective:.3e}",
                )
            )
            # Diffusive CFL: dt <= dx^2 / (2 * nu)
            nu_param = problem.get_parameter("nu")
            nu_val = nu_param.value if (nu_param and nu_param.value) else 0.01
            dt_diffusive = (dx_val ** 2) / (2.0 * nu_val)
            bounds.append(
                CFLStabilityBound(
                    cfl_type="diffusive",
                    condition_str="Δt ≤ Δx² / (2ν)",
                    max_dt_formula="dx^2 / (2 * nu)",
                    numerical_max_dt=dt_diffusive,
                    notes=f"Diffusive stability (explicit): ν={nu_val:.4f} -> Δt ≤ {dt_diffusive:.3e}",
                )
            )

        elif problem.id == "allen-cahn":
            # Allen-Cahn has diffusion eps^2 and stiff cubic reaction f(u) = u^3 - u
            eps_param = problem.get_parameter("eps")
            eps_val = eps_param.value if (eps_param and eps_param.value) else 0.05
            dt_diffusive = (dx_val ** 2) / (2.0 * (eps_val ** 2))
            bounds.append(
                CFLStabilityBound(
                    cfl_type="diffusive",
                    condition_str="Δt ≤ Δx² / (2ε²)",
                    max_dt_formula="dx^2 / (2 * eps^2)",
                    numerical_max_dt=dt_diffusive,
                    notes=f"Interfacial diffusion limit: ε={eps_val:.4f} -> Δt ≤ {dt_diffusive:.3e}",
                )
            )
            # Reaction stiff limit: |f'(u)| <= 2 -> dt <= 2 / |f'|
            dt_reaction = 1.0
            bounds.append(
                CFLStabilityBound(
                    cfl_type="reaction",
                    condition_str="Δt ≤ 2 / max|f'(u)| = 1.0",
                    max_dt_formula="2 / max|3u^2 - 1|",
                    numerical_max_dt=dt_reaction,
                    notes="Reaction stiffness: IMEX split-step avoids explicit restriction.",
                )
            )

        elif problem.id == "nlse":
            # Dispersive split-step phase limit: delta_phi = gamma * |psi|^2 * dt << pi/2
            gamma_param = problem.get_parameter("gamma")
            gamma_val = gamma_param.value if (gamma_param and gamma_param.value) else 1.0
            dt_dispersive = 0.1 / (gamma_val * (u_scale ** 2))
            bounds.append(
                CFLStabilityBound(
                    cfl_type="dispersive",
                    condition_str="Δt ≪ 1 / (γ·max|ψ|²)",
                    max_dt_formula="0.1 / (gamma * max|psi|^2)",
                    numerical_max_dt=dt_dispersive,
                    notes=f"Split-step phase rotation < 0.1 rad: max|ψ|={u_scale:.2f} -> Δt ≤ {dt_dispersive:.3e}",
                )
            )

        elif problem.id == "duffing":
            # ODE oscillator stability: dt < 2 / omega_max or 1 / (10 * omega_driving)
            omega_param = problem.get_parameter("omega")
            omega_val = omega_param.value if (omega_param and omega_param.value) else 1.2
            dt_ode = (2.0 * 3.1415926535) / (20.0 * omega_val)
            bounds.append(
                CFLStabilityBound(
                    cfl_type="reaction",
                    condition_str="Δt ≤ T_period / 20 = π / (10ω)",
                    max_dt_formula="pi / (10 * omega)",
                    numerical_max_dt=dt_ode,
                    notes=f"Driving period T={2.0*3.14159/omega_val:.2f}s -> recommended Δt ≤ {dt_ode:.3e}",
                )
            )

        return bounds


# ---------------------------------------------------------------------------
# Typst Math Syntax Conversion
# ---------------------------------------------------------------------------

def to_typst_math(expr_or_latex: Union[sp.Expr, str]) -> str:
    """Convert a SymPy expression or LaTeX equation string into clean Typst math markup.

    Typst math syntax:
      - Enclosed between '$' or formatted for a block math ($ ... $).
      - Fractions: frac(a, b) or (a) / (b).
      - Partial derivatives: (diff u) / (diff t) or frac(diff u, diff x).
      - Powers: x^2.
      - Greek letters: nu, epsilon, gamma, delta, alpha, beta, omega.
      - Multiplication: thin space or center-dot dots.
    """
    if isinstance(expr_or_latex, sp.Expr):
        # Convert SymPy expression to LaTeX first
        latex_str = sp.latex(expr_or_latex)
    else:
        latex_str = str(expr_or_latex).strip()

    # Strip existing enclosing $
    if latex_str.startswith("$") and latex_str.endswith("$"):
        latex_str = latex_str[1:-1].strip()

    s = latex_str

    # Clean double derivatives
    # \frac{\partial^2 u}{\partial x^2} -> (partial^2 u) / (partial x^2)
    s = re.sub(
        r"\\frac\{\\partial(?:\^\{?2\}?)\s*([a-zA-Z]+)\}\{\\partial\s*([a-zA-Z])(?:\^\{?2\}?)\}",
        r"(partial^2 \1) / (partial \2^2)",
        s,
    )
    s = re.sub(
        r"\\frac\{\\partial\s*([a-zA-Z]+)\}\{\\partial\s*([a-zA-Z])\}",
        r"(partial \1) / (partial \2)",
        s,
    )

    # Trig and special functions: insert space before function name if adjacent to symbol
    s = re.sub(r"([a-zA-Z0-9_]+)\s*\\cos", r"\1 cos", s)
    s = re.sub(r"([a-zA-Z0-9_]+)\s*\\sin", r"\1 sin", s)
    s = re.sub(r"([a-zA-Z0-9_]+)\s*\\exp", r"\1 exp", s)

    # Fractions: \frac{a}{b} -> (a) / (b)
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1) / (\2)", s)
    s = re.sub(r"\\frac\(([^()]+)\)\(([^()]+)\)", r"(\1) / (\2)", s)
    s = s.replace(r"\frac", "")

    # Greek and tokens
    greek_replacements = {
        r"\nu": "nu",
        r"\varepsilon": "epsilon",
        r"\eps": "epsilon",
        r"\gamma": "gamma",
        r"\delta": "delta",
        r"\alpha": "alpha",
        r"\beta": "beta",
        r"\omega": "omega",
        r"\psi": "psi",
        r"\pi": "pi",
        r"\partial": "partial",
        r"\ddot": "dot.double",
        r"\dot": "dot",
        r"\cos": "cos",
        r"\sin": "sin",
    }
    for k, v in greek_replacements.items():
        s = s.replace(k, v)

    # Dots on variables: dot.double{x} -> dot.double(x)
    s = re.sub(r"dot\.double\{([a-zA-Z]+)\}", r"dot.double(\1)", s)
    s = re.sub(r"dot\{([a-zA-Z]+)\}", r"dot(\1)", s)

    # Cleanup braces
    s = s.replace("{", "(").replace("}", ")")
    # Clean LaTeX spacing
    s = s.replace(r"\,", " ").replace(r"\;", " ").replace(r"\quad", " ")
    s = s.replace(r"\left", "").replace(r"\right", "")

    # Trig and special functions: insert space before function name if adjacent to symbol
    s = re.sub(r"([a-zA-Z0-9_]+)\s*\\cos", r"\1 cos", s)
    s = re.sub(r"([a-zA-Z0-9_]+)\s*\\sin", r"\1 sin", s)
    s = re.sub(r"([a-zA-Z0-9_]+)\s*\\exp", r"\1 exp", s)
    # Format cleanly inside $ ... $
    clean_typst = " ".join(s.split())
    return f"$ {clean_typst} $"
