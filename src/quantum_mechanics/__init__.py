"""
Quantum Mechanics Emergence Layer for Intrinsic Resonance Holography v21.0

THEORETICAL FOUNDATION: IRH21.md §5, Appendix I

This module derives quantum mechanical phenomena from the wave interference
structure of the cGFT condensate. The Born rule, measurement, and decoherence
all emerge from the fundamental algorithmic information dynamics.

Key Results:
    - Appendix I.1: Emergent Hilbert space from wave interference
    - Appendix I.2: Born rule derivation from Algorithmic Selection
    - Appendix I.3: Lindblad equation for decoherence
    - §5.2.1: Quantifiable observer back-reaction

Modules:
    born_rule: Born rule, Lindblad equation, decoherence, measurement
    hilbert_space: Emergent ℋ from wave interference (Appendix I.1)
    entanglement: Quantum correlations from QNCD

Dependencies:
    - src.primitives (Layer 0)
    - src.cgft (Layer 1)
    - src.rg_flow (Layer 2)
    - src.emergent_spacetime (Layer 3)

Authors: IRH Computational Framework Team
Last Updated: 2024-12 (synchronized with IRH21.md v21.0)
"""

__version__ = "21.0.0"
__theoretical_foundation__ = "IRH21.md §5, Appendix I"

# Import from born_rule module
from .born_rule import (
    BornRule,
    DecoherenceRate,
    LindbladEquation,
    PointerBasis,
    MeasurementResolution,
    EmergentHilbertSpace,
    QuantumMechanicsEmergence,
    derive_born_rule,
    compute_decoherence_rate,
    derive_lindblad_equation,
    compute_pointer_basis,
    resolve_measurement_problem,
    derive_hilbert_space,
    compute_qm_emergence,
    compute_decoherence_time_estimate,
    verify_qm_emergence,
)

__all__ = [
    # Classes
    'BornRule',
    'DecoherenceRate',
    'LindbladEquation',
    'PointerBasis',
    'MeasurementResolution',
    'EmergentHilbertSpace',
    'QuantumMechanicsEmergence',
    
    # born_rule exports
    'derive_born_rule',
    'compute_decoherence_rate',
    'derive_lindblad_equation',
    'compute_pointer_basis',
    'resolve_measurement_problem',
    'derive_hilbert_space',
    'compute_qm_emergence',
    'compute_decoherence_time_estimate',
    'verify_qm_emergence',
]
