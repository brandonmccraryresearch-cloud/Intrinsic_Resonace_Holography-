"""
Engines module for symbolic reasoning and field dynamics.

This module provides the symbolic computation engines for IRH,
adapted from AlphaGeometry's DD+AR reasoning system.
"""

from .holographic_state import CouplingState, HolographicState
from .resonance_engine import ResonanceEngine

__all__ = [
    'CouplingState',
    'HolographicState',
    'ResonanceEngine',
]
