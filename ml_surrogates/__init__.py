"""
ML Surrogates for Intrinsic Resonance Holography (IRH)

This module provides machine learning surrogate models for accelerating
IRH computations, adapted from the AlphaGeometry architecture.

THEORETICAL FOUNDATION: IRH v21.1 §1.2-1.3

Main components:
- engines: Symbolic computation engines for field dynamics
- models: Transformer-based neural network architectures
- training: Training infrastructure and data loading
- utils: Utility functions for graph conversion and visualization
"""

from ml_surrogates.engines import CouplingState, HolographicState
from ml_surrogates.engines import ResonanceEngine

__version__ = "0.1.0"
__all__ = [
    "CouplingState",
    "HolographicState",
    "ResonanceEngine",
]
