"""
Cosmology Layer for Intrinsic Resonance Holography v21.0

THEORETICAL FOUNDATION: IRH21.md §2.3

This module derives cosmological predictions from the cGFT fixed point,
including the cosmological constant from the "Holographic Hum" and the
dark energy equation of state w(z).

Key Results:
    - Eq. 2.17-2.19: ρ_hum calculation (Holographic Hum)
    - Eq. 2.21-2.23: w(z) equation of state
    - Appendix C.6-C.8: Running fundamental constants c(k), ℏ(k), G(k)

Modules:
    dark_energy: Dark energy EoS w₀, Holographic Hum, vacuum energy
    running_constants: c(k), ℏ(k), G(k) (Appendix C.6-C.8)
    primordial_universe: Early universe, inflation signatures

Dependencies:
    - src.primitives (Layer 0)
    - src.cgft (Layer 1)
    - src.rg_flow (Layer 2)
    - src.emergent_spacetime (Layer 3)
    - src.topology (Layer 4)
    - src.standard_model (Layer 5)

Authors: IRH Computational Framework Team
Last Updated: 2024-12 (synchronized with IRH21.md v21.0)
"""

__version__ = "21.0.0"
__theoretical_foundation__ = "IRH21.md §2.3"

# Import from dark_energy module
from .dark_energy import (
    HolographicHum,
    DarkEnergyEoS,
    VacuumEnergyCancellation,
    CosmologicalConstant,
    W0_PREDICTION,
    W0_UNCERTAINTY,
    compute_holographic_hum,
    compute_dark_energy_eos,
    compute_dark_energy_density,
    compute_vacuum_energy_cancellation,
    compute_cosmological_constant,
    verify_dark_energy_predictions,
    compute_hubble_tension_resolution,
)

# Predicted dark energy equation of state (Eq. 2.23)
W_0_PREDICTED = W0_PREDICTION  # w₀ with 8-digit precision

__all__ = [
    # Constants
    'W_0_PREDICTED',
    'W0_PREDICTION',
    'W0_UNCERTAINTY',
    
    # Classes
    'HolographicHum',
    'DarkEnergyEoS',
    'VacuumEnergyCancellation',
    'CosmologicalConstant',
    
    # dark_energy exports
    'compute_holographic_hum',
    'compute_dark_energy_eos',
    'compute_dark_energy_density',
    'compute_vacuum_energy_cancellation',
    'compute_cosmological_constant',
    'verify_dark_energy_predictions',
    'compute_hubble_tension_resolution',
]
