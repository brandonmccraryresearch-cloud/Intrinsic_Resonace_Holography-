"""
IRH Desktop - Transparency Package

Provides the transparency engine for verbose, contextual output.
All computations explain themselves with theoretical references.

Author: Brandon D. McCrary
"""

from irh_desktop.transparency.engine import (
    TransparencyEngine,
    TransparentMessage,
    MessageLevel,
)

__all__ = [
    "TransparencyEngine",
    "TransparentMessage",
    "MessageLevel",
]
