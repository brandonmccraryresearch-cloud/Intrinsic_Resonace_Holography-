"""
Falsifiable Predictions Layer for Intrinsic Resonance Holography v21.0

THEORETICAL FOUNDATION: IRH21.md §8, Appendix J

This module extracts testable, falsifiable predictions that connect the
mathematical formalism to experimental reality. This is the "tip of the
iceberg" where IRH confronts Nature's tribunal.

Key Predictions:
    - §2.5, Eq. 2.24-2.26: Lorentz Invariance Violation parameter ξ
    - Appendix J.1: Generation-specific LIV thresholds
    - Appendix J.2: Gravitational wave sidebands
    - Appendix J.3: Muon g-2 anomaly resolution
    - Appendix J.4: Higgs trilinear coupling λ_HHH
    - §5.2.1: Quantifiable observer back-reaction

Falsification Timeline (§8.7):
    - 2029: JUNO neutrino hierarchy (test normal hierarchy prediction)
    - 2029: Euclid/Roman dark energy constraints (test w₀ = -0.912)
    - 2029: CTA Lorentz invariance violation bounds (test ξ = 1.93×10⁻⁴)

Modules:
    lorentz_violation: LIV parameter ξ (Eq. 2.24-2.26)
    muon_g_minus_2: Anomalous magnetic moment (Appendix J.3)
    gravitational_sidebands: GW sidebands from discrete spacetime (Appendix J.2)

Dependencies:
    - All previous layers (0-7)

Authors: IRH Computational Framework Team
Last Updated: 2024-12 (synchronized with IRH21.md v21.0)
"""

__version__ = "21.0.0"
__theoretical_foundation__ = "IRH21.md §8, Appendix J"

# Import from lorentz_violation module
from .lorentz_violation import (
    LIVParameter,
    ModifiedDispersion,
    GenerationLIV,
    PhotonDispersion,
    GammaRayTest,
    XI_LIV,
    XI_CERTIFIED,
    compute_liv_parameter,
    compute_modified_dispersion,
    compute_generation_liv,
    compute_photon_time_delay,
    analyze_grb_liv_test,
    verify_liv_predictions,
    compute_cta_sensitivity,
)

# Import from muon_g_minus_2 module
from .muon_g_minus_2 import (
    MuonAnomalousMMResult,
    A_MU_EXPERIMENTAL,
    A_MU_SM,
    A_MU_ANOMALY,
    compute_qed_contribution,
    compute_irh_vwp_contribution,
    compute_muon_g_minus_2,
    analyze_anomaly_resolution,
    verify_muon_g2_predictions,
)

# Import from gravitational_sidebands module
from .gravitational_sidebands import (
    GWSideband,
    GWDetectorSensitivity,
    SpacetimeGranularity,
    DETECTORS,
    compute_gw_sidebands,
    analyze_detectability,
    compute_spacetime_granularity,
    predict_binary_merger_sidebands,
    verify_gw_sideband_predictions,
)

# Muon g-2 anomaly prediction (Eq. J.1)
DELTA_G_MINUS_2_MUON = A_MU_ANOMALY

# Higgs trilinear coupling prediction (Eq. J.2)
LAMBDA_HHH = 125.25  # GeV, ± 1.25 GeV

__all__ = [
    # Constants
    'XI_LIV',
    'XI_CERTIFIED',
    'DELTA_G_MINUS_2_MUON',
    'LAMBDA_HHH',
    'A_MU_EXPERIMENTAL',
    'A_MU_SM',
    'A_MU_ANOMALY',
    'DETECTORS',
    
    # Classes
    'LIVParameter',
    'ModifiedDispersion',
    'GenerationLIV',
    'PhotonDispersion',
    'GammaRayTest',
    'MuonAnomalousMMResult',
    'GWSideband',
    'GWDetectorSensitivity',
    'SpacetimeGranularity',
    
    # lorentz_violation exports
    'compute_liv_parameter',
    'compute_modified_dispersion',
    'compute_generation_liv',
    'compute_photon_time_delay',
    'analyze_grb_liv_test',
    'verify_liv_predictions',
    'compute_cta_sensitivity',
    
    # muon_g_minus_2 exports
    'compute_qed_contribution',
    'compute_irh_vwp_contribution',
    'compute_muon_g_minus_2',
    'analyze_anomaly_resolution',
    'verify_muon_g2_predictions',
    
    # gravitational_sidebands exports
    'compute_gw_sidebands',
    'analyze_detectability',
    'compute_spacetime_granularity',
    'predict_binary_merger_sidebands',
    'verify_gw_sideband_predictions',
]
