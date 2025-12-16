"""
IRH Desktop Application

Intrinsic Resonance Holography v21.0 Desktop Interface

A feature-rich desktop application providing:
- Transparent, verbose computation output
- Interactive visualization of physics derivations
- Auto-update system for IRH engine
- Customizable configuration profiles
- Plugin system for extensions

Theoretical Foundation:
    IRH21.md - Complete unified theory implementation

Author: Brandon D. McCrary
License: MIT
"""

__version__ = "21.0.0"
__author__ = "Brandon D. McCrary"

from irh_desktop.app import IRHDesktopApp
from irh_desktop.core.engine_manager import EngineManager
from irh_desktop.core.config_manager import ConfigManager
from irh_desktop.transparency.engine import TransparencyEngine

__all__ = [
    "IRHDesktopApp",
    "EngineManager",
    "ConfigManager",
    "TransparencyEngine",
    "__version__",
]
