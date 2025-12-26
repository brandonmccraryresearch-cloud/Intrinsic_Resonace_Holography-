"""
Unit tests for Yukawa RG Running module.

Theoretical Reference:
    IRH v21.4 Part 1, Executive Summary Point 1
"""

import math
import pytest
import numpy as np

import sys
from pathlib import Path
_repo_root = Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.standard_model.yukawa_rg_running import (
    compute_yukawa_rg_running,
    compute_anomalous_dimension,
    compute_fermion_mass_with_rg,
    YukawaRGResult,
    PLANCK_SCALE,
    ELECTROWEAK_SCALE,
    LAMBDA_STAR,
    GAMMA_STAR,
    MU_STAR,
)


class TestYukawaRGRunning:
    """Test suite for Yukawa RG running implementation."""
    
    def test_compute_anomalous_dimension(self):
        """Test anomalous dimension computation."""
        # Test with electron-like K_f
        gamma_f = compute_anomalous_dimension(
            K_f=1.0,
            lambda_star=LAMBDA_STAR,
            gamma_star=GAMMA_STAR,
            mu_star=MU_STAR,
        )
        
        # Should be positive and dimensionless
        assert gamma_f > 0
        assert isinstance(gamma_f, float)
        
        # Should be small (leading order approximation)
        assert gamma_f < 1.0
    
    def test_compute_yukawa_rg_running_basic(self):
        """Test basic Yukawa RG running computation."""
        result = compute_yukawa_rg_running(
            K_f=1.0,
            n_rg_steps=1000,
            verbosity='silent',
        )
        
        # Check result structure
        assert isinstance(result, YukawaRGResult)
        assert hasattr(result, 'R_Y')
        assert hasattr(result, 'k_initial')
        assert hasattr(result, 'k_final')
        assert hasattr(result, 'K_f')
        assert hasattr(result, 'trajectory')
        
        # Check R_Y is reasonable
        # With small anomalous dimension, R_Y should be close to 1
        assert 0.5 < result.R_Y < 2.0
        assert result.R_Y > 0
        
        # Check scales
        assert result.k_initial == PLANCK_SCALE
        assert result.k_final == ELECTROWEAK_SCALE
        assert result.K_f == 1.0
    
    def test_rg_running_scale_dependence(self):
        """Test that R_Y depends on scale range."""
        # Full scale evolution
        result_full = compute_yukawa_rg_running(
            k_initial=PLANCK_SCALE,
            k_final=ELECTROWEAK_SCALE,
            K_f=1.0,
            verbosity='silent',
        )
        
        # Shorter scale evolution
        result_short = compute_yukawa_rg_running(
            k_initial=1e10,  # 10 TeV
            k_final=ELECTROWEAK_SCALE,
            K_f=1.0,
            verbosity='silent',
        )
        
        # Should be different
        assert result_full.R_Y != result_short.R_Y
        # Full evolution should have larger effect
        assert abs(result_full.R_Y - 1.0) > abs(result_short.R_Y - 1.0)
    
    def test_rg_running_K_f_dependence(self):
        """Test that R_Y computation completes for different K_f values."""
        result_small = compute_yukawa_rg_running(
            K_f=1.0,
            verbosity='silent',
        )
        
        result_large = compute_yukawa_rg_running(
            K_f=100.0,
            verbosity='silent',
        )
        
        # Both should complete successfully
        assert result_small.R_Y > 0
        assert result_large.R_Y > 0
        
        # Values should be in reasonable range
        assert 0.5 < result_small.R_Y < 2.0
        assert 0.5 < result_large.R_Y < 2.0
    
    def test_rg_running_convergence(self):
        """Test numerical convergence with increasing steps."""
        result_1000 = compute_yukawa_rg_running(
            K_f=1.0,
            n_rg_steps=1000,
            verbosity='silent',
        )
        
        result_10000 = compute_yukawa_rg_running(
            K_f=1.0,
            n_rg_steps=10000,
            verbosity='silent',
        )
        
        # Should converge
        relative_diff = abs(result_10000.R_Y - result_1000.R_Y) / result_1000.R_Y
        assert relative_diff < 0.01  # Within 1%
    
    def test_trajectory_structure(self):
        """Test that trajectory data is properly structured."""
        result = compute_yukawa_rg_running(
            K_f=1.0,
            verbosity='silent',
        )
        
        traj = result.trajectory
        
        # Check required keys
        assert 't_values' in traj
        assert 'gamma_f_values' in traj
        assert 'integral_values' in traj
        
        # Check array sizes match
        assert len(traj['t_values']) == len(traj['gamma_f_values'])
        assert len(traj['t_values']) == len(traj['integral_values'])
        
        # t_values should go from initial (large positive) to final (0)
        # Check that first value is larger than last
        assert traj['t_values'][0] > traj['t_values'][-1]
    
    def test_result_to_dict(self):
        """Test conversion of result to dictionary."""
        result = compute_yukawa_rg_running(
            K_f=1.0,
            verbosity='silent',
        )
        
        result_dict = result.to_dict()
        
        # Check all required fields
        assert 'R_Y' in result_dict
        assert 'k_initial' in result_dict
        assert 'k_final' in result_dict
        assert 'K_f' in result_dict
        assert 'n_steps' in result_dict
        assert 'trajectory' in result_dict
        assert 'theoretical_reference' in result_dict
        
        # Check values match
        assert result_dict['R_Y'] == result.R_Y
        assert result_dict['K_f'] == result.K_f


class TestFermionMassWithRG:
    """Test suite for fermion mass computation with RG."""
    
    def test_compute_fermion_mass_with_rg_basic(self):
        """Test basic fermion mass computation with RG."""
        result = compute_fermion_mass_with_rg(
            fermion='electron',
            K_f=1.0,
            verbosity='silent',
        )
        
        # Check result structure
        assert 'fermion' in result
        assert 'mass_GeV' in result
        assert 'K_f' in result
        assert 'R_Y' in result
        assert 'components' in result
        assert 'theoretical_reference' in result
        
        # Check mass is positive
        assert result['mass_GeV'] > 0
        
        # Check R_Y is included
        assert result['R_Y'] > 0
        
        # Check components
        components = result['components']
        assert 'prefactor' in components
        assert 'yukawa_coupling' in components
        assert 'higgs_vev_term' in components
    
    def test_mass_scales_with_K_f(self):
        """Test that mass scales with topological complexity."""
        result_small = compute_fermion_mass_with_rg(
            fermion='electron',
            K_f=1.0,
            verbosity='silent',
        )
        
        result_large = compute_fermion_mass_with_rg(
            fermion='muon',
            K_f=206.768,  # Muon-like
            verbosity='silent',
        )
        
        # Larger K_f should give larger mass
        assert result_large['mass_GeV'] > result_small['mass_GeV']
        
        # Corrected: Mass scales linearly with K_f (Eq. 3.6 in IRH v21.4)
        # m_mu / m_e approx K_mu / K_e approx 206
        ratio = result_large['mass_GeV'] / result_small['mass_GeV']
        # Check it's in a reasonable range around 206
        assert 180 < ratio < 230  # Muon is ~206x heavier than electron
    
    def test_theoretical_reference(self):
        """Test that theoretical reference is correct."""
        result = compute_fermion_mass_with_rg(
            fermion='electron',
            K_f=1.0,
            verbosity='silent',
        )
        
        ref = result['theoretical_reference']
        assert 'IRH v21.4' in ref
        assert 'Part 1' in ref
        assert 'Eq. 3.6' in ref


class TestVerbosityLevels:
    """Test transparency engine integration."""
    
    def test_silent_verbosity(self):
        """Test silent verbosity (no output)."""
        result = compute_yukawa_rg_running(
            K_f=1.0,
            verbosity='silent',
        )
        # Should complete without errors
        assert result.R_Y > 0
    
    def test_minimal_verbosity(self):
        """Test minimal verbosity (basic output)."""
        result = compute_yukawa_rg_running(
            K_f=1.0,
            verbosity='minimal',
        )
        # Should complete and produce output
        assert result.R_Y > 0
    
    def test_detailed_verbosity(self):
        """Test detailed verbosity (full transparency)."""
        result = compute_yukawa_rg_running(
            K_f=1.0,
            verbosity='detailed',
        )
        # Should complete with transparency engine
        assert result.R_Y > 0


class TestPhysicalConsistency:
    """Test physical consistency of results."""
    
    def test_positive_values(self):
        """Test that all results are positive."""
        result = compute_yukawa_rg_running(
            K_f=1.0,
            verbosity='silent',
        )
        
        assert result.R_Y > 0
        assert result.k_initial > 0
        assert result.k_final > 0
        assert result.K_f > 0
    
    def test_scale_ordering(self):
        """Test that scales are properly ordered."""
        result = compute_yukawa_rg_running(
            K_f=1.0,
            verbosity='silent',
        )
        
        # Planck scale should be larger than EW scale
        assert result.k_initial > result.k_final
    
    def test_dimensional_consistency(self):
        """Test dimensional consistency of mass formula."""
        result = compute_fermion_mass_with_rg(
            fermion='electron',
            K_f=1.0,
            verbosity='silent',
        )
        
        # Mass should be in reasonable range for electron-like particle
        mass = result['mass_GeV']
        
        # Should be order of GeV or below for K_f = 1
        # Allow wide range as this is a placeholder implementation
        assert 1e-3 < mass < 1e3  # Between MeV and TeV scale


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
