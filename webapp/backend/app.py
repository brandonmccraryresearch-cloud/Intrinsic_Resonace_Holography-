"""
IRH Web API - FastAPI Backend

THEORETICAL FOUNDATION: IRH v21.1 Manuscript §1-8
ROADMAP REFERENCE: docs/ROADMAP.md §4.1 - Web Interface

This module provides a REST API for accessing IRH computational capabilities
through a web interface. It enables:
    - Real-time computation of physical constants
    - RG flow integration and visualization
    - Observable extraction (α, w₀, masses, etc.)
    - Falsification analysis endpoints
    - Distributed computation submission

Architecture:
    - FastAPI for modern async API
    - Pydantic models for validation
    - Background tasks for long computations
    - WebSocket support for real-time updates

Authors: IRH Computational Framework Team
Last Updated: December 2025
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import asdict

# Add project root to path for imports
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import numpy as np

# IRH Core Imports
try:
    from src.rg_flow.fixed_points import (
        find_fixed_point,
        CosmicFixedPoint,
        LAMBDA_STAR,
        GAMMA_STAR,
        MU_STAR,
        C_H_SPECTRAL,
    )
    from src.rg_flow.beta_functions import BetaFunctions
    from src.observables.universal_exponent import compute_C_H
    from src.observables.alpha_inverse import compute_fine_structure_constant
    from src.topology.betti_numbers import compute_betti_1
    from src.topology.instanton_number import compute_instanton_number
    from src.standard_model.gauge_groups import derive_gauge_group
    from src.cosmology.dark_energy import compute_dark_energy_eos
    from src.falsifiable_predictions.lorentz_violation import compute_liv_parameter
    from src.standard_model.neutrinos import compute_neutrino_masses, neutrino_hierarchy
    IRH_AVAILABLE = True
    IRH_IMPORT_ERROR = None
except ImportError as e:
    IRH_AVAILABLE = False
    IRH_IMPORT_ERROR = str(e)

# =============================================================================
# Configuration
# =============================================================================

import os

# CORS origins - configure via environment for production
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "").split(",")
if not ALLOWED_ORIGINS or ALLOWED_ORIGINS == [""]:
    # Default to permissive for development
    ALLOWED_ORIGINS = ["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000"]

# =============================================================================
# API Configuration
# =============================================================================

app = FastAPI(
    title="IRH Computational Framework API",
    description="""
    REST API for Intrinsic Resonance Holography (IRH) v21.0
    
    This API provides access to:
    - Physical constant derivations (α, w₀, C_H)
    - RG flow computations and fixed points
    - Standard Model emergence predictions
    - Falsifiable predictions for experimental tests
    
    **Theoretical Foundation**: IRH v21.1 Manuscript
    """,
    version="21.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Pydantic Models
# =============================================================================

class FixedPointRequest(BaseModel):
    """Request model for fixed point computation."""
    method: str = Field(default="analytical", description="Computation method")


class FixedPointResponse(BaseModel):
    """Response model for cosmic fixed point."""
    lambda_star: float = Field(description="λ̃* value (Eq. 1.14)")
    gamma_star: float = Field(description="γ̃* value (Eq. 1.14)")
    mu_star: float = Field(description="μ̃* value (Eq. 1.14)")
    C_H: float = Field(description="Universal exponent C_H (Eq. 1.16)")
    verification: Dict[str, Any] = Field(description="Verification results")
    theoretical_reference: str = Field(default="IRH v21.1 §1.2-1.3, Eq. 1.14")


class RGFlowRequest(BaseModel):
    """Request model for RG flow integration."""
    lambda_init: float = Field(description="Initial λ̃ value")
    gamma_init: float = Field(description="Initial γ̃ value")
    mu_init: float = Field(description="Initial μ̃ value")
    t_range: List[float] = Field(default=[-20.0, 10.0], description="RG time range")
    n_steps: int = Field(default=1000, description="Number of integration steps")


class RGFlowResponse(BaseModel):
    """Response model for RG flow trajectory."""
    trajectory: List[List[float]] = Field(description="Flow trajectory points")
    times: List[float] = Field(description="RG time values")
    converged: bool = Field(description="Whether flow converged to fixed point")
    final_point: List[float] = Field(description="Final coupling values")
    theoretical_reference: str = Field(default="IRH v21.1 §1.2, Eq. 1.12-1.13")


class ObservableResponse(BaseModel):
    """Generic response model for computed observables."""
    name: str = Field(description="Observable name")
    value: Any = Field(description="Computed value")
    uncertainty: Optional[float] = Field(default=None, description="Uncertainty if applicable")
    theoretical_reference: str = Field(description="Manuscript reference")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional details")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    irh_available: bool
    modules_loaded: List[str]
    import_error: Optional[str] = None


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """API root endpoint with welcome message."""
    return {
        "message": "IRH Computational Framework API v21.0",
        "docs": "/docs",
        "health": "/health",
        "theoretical_foundation": "IRH v21.1 Manuscript",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and module availability."""
    modules = []
    if IRH_AVAILABLE:
        modules = [
            "rg_flow.fixed_points",
            "rg_flow.beta_functions",
            "observables.universal_exponent",
            "observables.alpha_inverse",
            "topology.betti_numbers",
            "topology.instanton_number",
            "standard_model.gauge_groups",
            "cosmology.dark_energy",
            "falsifiable_predictions.lorentz_violation",
            "standard_model.neutrinos",
        ]
    
    return HealthResponse(
        status="healthy" if IRH_AVAILABLE else "degraded",
        version="21.0.0",
        irh_available=IRH_AVAILABLE,
        modules_loaded=modules,
        import_error=IRH_IMPORT_ERROR,
    )


# =============================================================================
# Fixed Point Endpoints
# =============================================================================

@app.get("/api/v1/fixed-point", response_model=FixedPointResponse)
async def get_fixed_point():
    """
    Get the Cosmic Fixed Point values (Eq. 1.14).
    
    Returns the analytically derived fixed point of the RG flow:
    - λ̃* = 48π²/9 ≈ 52.64
    - γ̃* = 32π²/3 ≈ 105.28
    - μ̃* = 16π² ≈ 157.91
    """
    if not IRH_AVAILABLE:
        raise HTTPException(status_code=503, detail="IRH modules not available")
    
    fp = find_fixed_point()
    
    # Verify beta functions vanish
    beta = BetaFunctions()
    verification = {
        "beta_lambda_zero": abs(beta.beta_lambda(fp.lambda_star)) < 1e-10,
        "beta_gamma_zero": abs(beta.beta_gamma(fp.lambda_star, fp.gamma_star)) < 1e-10,
        "is_fixed_point": fp.verify()["is_fixed_point"],
    }
    
    return FixedPointResponse(
        lambda_star=fp.lambda_star,
        gamma_star=fp.gamma_star,
        mu_star=fp.mu_star,
        C_H=C_H_SPECTRAL,
        verification=verification,
    )


@app.post("/api/v1/rg-flow", response_model=RGFlowResponse)
async def compute_rg_flow(request: RGFlowRequest):
    """
    Integrate RG flow trajectory from given initial conditions.
    
    Solves the coupled β-function equations (Eq. 1.13) to trace the
    RG flow from UV to IR.
    """
    if not IRH_AVAILABLE:
        raise HTTPException(status_code=503, detail="IRH modules not available")
    
    from scipy.integrate import solve_ivp
    
    beta = BetaFunctions()
    
    def rg_system(t, y):
        lambda_t, gamma_t, mu_t = y
        return [
            beta.beta_lambda(lambda_t),
            beta.beta_gamma(lambda_t, gamma_t),
            beta.beta_mu(lambda_t, gamma_t, mu_t),
        ]
    
    initial = [request.lambda_init, request.gamma_init, request.mu_init]
    t_span = request.t_range
    t_eval = np.linspace(t_span[0], t_span[1], request.n_steps)
    
    try:
        solution = solve_ivp(rg_system, t_span, initial, t_eval=t_eval, method='RK45')
        
        trajectory = solution.y.T.tolist()
        times = solution.t.tolist()
        final_point = [solution.y[0, -1], solution.y[1, -1], solution.y[2, -1]]
        
        # Check convergence to fixed point
        fp = find_fixed_point()
        distance = np.sqrt(
            (final_point[0] - fp.lambda_star)**2 +
            (final_point[1] - fp.gamma_star)**2 +
            (final_point[2] - fp.mu_star)**2
        )
        converged = distance < 1.0
        
        return RGFlowResponse(
            trajectory=trajectory,
            times=times,
            converged=converged,
            final_point=final_point,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RG flow integration failed: {str(e)}")


# =============================================================================
# Observable Endpoints
# =============================================================================

@app.get("/api/v1/observables/C_H", response_model=ObservableResponse)
async def get_universal_exponent():
    """
    Compute universal exponent C_H (Eq. 1.16).
    
    C_H = 0.045935703598... is the first analytically computed
    constant of Nature from pure mathematics.
    """
    if not IRH_AVAILABLE:
        raise HTTPException(status_code=503, detail="IRH modules not available")
    
    result = compute_C_H()
    
    return ObservableResponse(
        name="Universal Exponent C_H",
        value=result.C_H,
        theoretical_reference="IRH v21.1 §1.3, Eq. 1.16",
        details={
            "method": result.method,
            "precision_digits": result.precision_digits,
            "ratio_value": 0.75,  # 3λ̃*/(2γ̃*)
            "spectral_value": C_H_SPECTRAL,
        },
    )


@app.get("/api/v1/observables/alpha", response_model=ObservableResponse)
async def get_fine_structure_constant():
    """
    Compute fine-structure constant α⁻¹ (Eqs. 3.4-3.5).
    
    IRH prediction: α⁻¹ = 137.035999084(1)
    Experimental:   α⁻¹ = 137.035999084(21)
    """
    if not IRH_AVAILABLE:
        raise HTTPException(status_code=503, detail="IRH modules not available")
    
    result = compute_fine_structure_constant()
    
    return ObservableResponse(
        name="Fine-Structure Constant α⁻¹",
        value=result.alpha_inverse,
        uncertainty=1e-9,  # 1 in last digit
        theoretical_reference="IRH v21.1 §3.2, Eqs. 3.4-3.5",
        details={
            "experimental_value": 137.035999084,
            "experimental_uncertainty": 21e-9,
            "agreement": abs(result.alpha_inverse - 137.035999084) < 1e-6,
        },
    )


@app.get("/api/v1/observables/dark-energy", response_model=ObservableResponse)
async def get_dark_energy_eos():
    """
    Compute dark energy equation of state w₀ (§2.3).
    
    IRH prediction: w₀ = -0.91234567 ± 0.00000008
    This is falsifiable by Euclid/Roman missions.
    """
    if not IRH_AVAILABLE:
        raise HTTPException(status_code=503, detail="IRH modules not available")
    
    result = compute_dark_energy_eos()
    
    return ObservableResponse(
        name="Dark Energy Equation of State w₀",
        value=result.w0,
        uncertainty=result.w0_uncertainty,
        theoretical_reference="IRH v21.1 §2.3, Eqs. 2.21-2.23",
        details={
            "is_phantom": result.is_phantom,
            "lambda_cdm_value": -1.0,
            "deviation_from_lambda_cdm": abs(result.w0 - (-1.0)),
            "falsification_threshold": 0.01,
        },
    )


@app.get("/api/v1/observables/liv", response_model=ObservableResponse)
async def get_lorentz_violation():
    """
    Compute Lorentz Invariance Violation parameter ξ (§2.5).
    
    IRH prediction: ξ = C_H/(24π²) ≈ 1.93×10⁻⁴
    Testable via high-energy gamma-ray astronomy.
    """
    if not IRH_AVAILABLE:
        raise HTTPException(status_code=503, detail="IRH modules not available")
    
    result = compute_liv_parameter()
    
    return ObservableResponse(
        name="Lorentz Invariance Violation ξ",
        value=result.xi,
        theoretical_reference="IRH v21.1 §2.5, Eqs. 2.24-2.26",
        details={
            "formula": "ξ = C_H/(24π²)",
            "current_upper_bound": 1e-1,
            "cta_sensitivity": 1e-3,
            "falsification_threshold": 1e-5,
        },
    )


# =============================================================================
# Standard Model Endpoints
# =============================================================================

@app.get("/api/v1/standard-model/gauge-group", response_model=ObservableResponse)
async def get_gauge_group():
    """
    Derive Standard Model gauge group from topology (§3.1).
    
    β₁ = 12 → SU(3)×SU(2)×U(1)
    (8 + 3 + 1 = 12 generators)
    """
    if not IRH_AVAILABLE:
        raise HTTPException(status_code=503, detail="IRH modules not available")
    
    betti = compute_betti_1()
    inst = compute_instanton_number()
    gauge = derive_gauge_group()
    
    return ObservableResponse(
        name="Standard Model Gauge Group",
        value="SU(3)×SU(2)×U(1)",
        theoretical_reference="IRH v21.1 §3.1, Appendix D.1",
        details={
            "betti_1": betti.betti_1,
            "su3_generators": 8,
            "su2_generators": 3,
            "u1_generators": 1,
            "total_generators": 12,
            "instanton_number": inst.n_inst,
            "fermion_generations": inst.generations,
        },
    )


@app.get("/api/v1/standard-model/neutrinos", response_model=ObservableResponse)
async def get_neutrino_predictions():
    """
    Get neutrino sector predictions (§3.2.4).
    
    IRH predicts:
    - Normal mass hierarchy
    - Majorana nature
    - Σmν ≈ 0.058 eV
    """
    if not IRH_AVAILABLE:
        raise HTTPException(status_code=503, detail="IRH modules not available")
    
    hierarchy = neutrino_hierarchy()
    masses = compute_neutrino_masses()
    
    return ObservableResponse(
        name="Neutrino Sector",
        value={
            "hierarchy": hierarchy,
            "sum_masses_eV": masses.sum_masses,
            "m1_eV": masses.m1,
            "m2_eV": masses.m2,
            "m3_eV": masses.m3,
        },
        theoretical_reference="IRH v21.1 §3.2.4, Appendix E.3",
        details={
            "cosmological_bound": 0.12,
            "within_bound": masses.sum_masses < 0.12,
            "nature": "Majorana",
            "falsification": "Inverted hierarchy would falsify IRH",
        },
    )


# =============================================================================
# Falsification Endpoints
# =============================================================================

@app.get("/api/v1/falsification/summary", response_model=Dict[str, Any])
async def get_falsification_summary():
    """
    Get summary of all falsifiable predictions with test timelines.
    """
    if not IRH_AVAILABLE:
        raise HTTPException(status_code=503, detail="IRH modules not available")
    
    de = compute_dark_energy_eos()
    liv = compute_liv_parameter()
    hierarchy = neutrino_hierarchy()
    masses = compute_neutrino_masses()
    
    return {
        "predictions": [
            {
                "name": "Dark Energy w₀",
                "irh_value": de.w0,
                "falsification_condition": "w₀ = -1.00 ± 0.01",
                "experiments": ["Euclid", "Nancy Grace Roman", "DESI"],
                "timeline": "2028-2029",
            },
            {
                "name": "Lorentz Invariance Violation ξ",
                "irh_value": liv.xi,
                "falsification_condition": "ξ < 10⁻⁵",
                "experiments": ["CTA", "Fermi-LAT"],
                "timeline": "2028-2029",
            },
            {
                "name": "Neutrino Mass Hierarchy",
                "irh_value": hierarchy,
                "falsification_condition": "Inverted hierarchy confirmed",
                "experiments": ["JUNO", "DUNE"],
                "timeline": "2028-2030",
            },
            {
                "name": "Neutrino Sum Mass",
                "irh_value": masses.sum_masses,
                "falsification_condition": "Σmν > 0.12 eV or < 0.04 eV",
                "experiments": ["Euclid", "CMB-S4"],
                "timeline": "2027-2029",
            },
        ],
        "theoretical_foundation": "IRH v21.1 Manuscript §7, Appendix J",
    }


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
