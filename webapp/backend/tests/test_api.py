"""
Tests for IRH Web API Backend

ROADMAP REFERENCE: docs/ROADMAP.md §4.1 - Web Interface

These tests verify the FastAPI endpoints for the IRH computational framework.
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add project root to path
_repo_root = Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from webapp.backend.app import app

client = TestClient(app)


# =============================================================================
# Basic Endpoint Tests
# =============================================================================

class TestBasicEndpoints:
    """Tests for basic API endpoints."""
    
    def test_root_endpoint(self):
        """Root endpoint should return welcome message."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "IRH" in data["message"]
    
    def test_health_endpoint(self):
        """Health endpoint should return status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert data["version"] == "21.0.0"


# =============================================================================
# Fixed Point Tests
# =============================================================================

class TestFixedPointEndpoints:
    """Tests for fixed point computation endpoints."""
    
    def test_get_fixed_point(self):
        """Should return Cosmic Fixed Point values."""
        response = client.get("/api/v1/fixed-point")
        
        # May fail if IRH modules not available
        if response.status_code == 503:
            pytest.skip("IRH modules not available")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check expected keys
        assert "lambda_star" in data
        assert "gamma_star" in data
        assert "mu_star" in data
        assert "C_H" in data
        
        # Check approximate values (Eq. 1.14)
        assert abs(data["lambda_star"] - 52.64) < 0.1
        assert abs(data["gamma_star"] - 105.28) < 0.1
        assert abs(data["mu_star"] - 157.91) < 0.1


# =============================================================================
# Observable Tests
# =============================================================================

class TestObservableEndpoints:
    """Tests for observable computation endpoints."""
    
    def test_get_universal_exponent(self):
        """Should return C_H value."""
        response = client.get("/api/v1/observables/C_H")
        
        if response.status_code == 503:
            pytest.skip("IRH modules not available")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["name"] == "Universal Exponent C_H"
        assert "value" in data
        assert "theoretical_reference" in data
    
    def test_get_fine_structure_constant(self):
        """Should return α⁻¹ value."""
        response = client.get("/api/v1/observables/alpha")
        
        if response.status_code == 503:
            pytest.skip("IRH modules not available")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "Fine-Structure" in data["name"]
        # Check value is close to 137
        assert abs(data["value"] - 137.036) < 0.001
    
    def test_get_dark_energy_eos(self):
        """Should return w₀ value."""
        response = client.get("/api/v1/observables/dark-energy")
        
        if response.status_code == 503:
            pytest.skip("IRH modules not available")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "Dark Energy" in data["name"]
        # Check w₀ is negative and > -1 (non-phantom)
        assert data["value"] < 0
        assert data["value"] > -1.0
    
    def test_get_lorentz_violation(self):
        """Should return LIV parameter ξ."""
        response = client.get("/api/v1/observables/liv")
        
        if response.status_code == 503:
            pytest.skip("IRH modules not available")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "Lorentz" in data["name"]
        # Check ξ is small but positive
        assert data["value"] > 0
        assert data["value"] < 1


# =============================================================================
# Standard Model Tests
# =============================================================================

class TestStandardModelEndpoints:
    """Tests for Standard Model endpoints."""
    
    def test_get_gauge_group(self):
        """Should return SU(3)×SU(2)×U(1) derivation."""
        response = client.get("/api/v1/standard-model/gauge-group")
        
        if response.status_code == 503:
            pytest.skip("IRH modules not available")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "SU(3)" in data["value"]
        assert data["details"]["betti_1"] == 12
        assert data["details"]["total_generators"] == 12
    
    def test_get_neutrino_predictions(self):
        """Should return neutrino sector predictions."""
        response = client.get("/api/v1/standard-model/neutrinos")
        
        if response.status_code == 503:
            pytest.skip("IRH modules not available")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check hierarchy is normal
        assert data["value"]["hierarchy"] == "normal"
        # Check sum of masses is within cosmological bound
        assert data["value"]["sum_masses_eV"] < 0.12


# =============================================================================
# Falsification Tests
# =============================================================================

class TestFalsificationEndpoints:
    """Tests for falsification summary endpoints."""
    
    def test_get_falsification_summary(self):
        """Should return all falsifiable predictions."""
        response = client.get("/api/v1/falsification/summary")
        
        if response.status_code == 503:
            pytest.skip("IRH modules not available")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "predictions" in data
        assert len(data["predictions"]) >= 4
        
        # Check structure of predictions
        for pred in data["predictions"]:
            assert "name" in pred
            assert "irh_value" in pred
            assert "experiments" in pred
            assert "timeline" in pred


# =============================================================================
# RG Flow Integration Tests
# =============================================================================

class TestRGFlowEndpoints:
    """Tests for RG flow integration endpoints."""
    
    def test_compute_rg_flow(self):
        """Should integrate RG flow from given initial conditions."""
        request_data = {
            "lambda_init": 20.0,
            "gamma_init": 50.0,
            "mu_init": 80.0,
            "t_range": [0, 5],
            "n_steps": 100,
        }
        
        response = client.post("/api/v1/rg-flow", json=request_data)
        
        if response.status_code == 503:
            pytest.skip("IRH modules not available")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "trajectory" in data
        assert "times" in data
        assert "converged" in data
        assert "final_point" in data
        
        # Check trajectory has some points (may be fewer due to adaptive stepping)
        assert len(data["times"]) >= 10
        assert len(data["trajectory"]) == len(data["times"])


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""
    
    def test_invalid_endpoint_returns_404(self):
        """Invalid endpoint should return 404."""
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404
    
    def test_invalid_rg_flow_request(self):
        """Invalid RG flow request should return error."""
        # Missing required fields
        response = client.post("/api/v1/rg-flow", json={})
        assert response.status_code == 422  # Validation error
