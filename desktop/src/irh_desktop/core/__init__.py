"""
IRH Desktop - Core Services Package

This package provides core services for the IRH Desktop application:
- Engine Manager: Manages IRH engine lifecycle
- Config Manager: Handles configuration files

Author: Brandon D. McCrary
"""

from irh_desktop.core.engine_manager import EngineManager, EngineInfo, UpdateInfo
from irh_desktop.core.config_manager import ConfigManager

__all__ = [
    "EngineManager",
    "EngineInfo",
    "UpdateInfo",
    "ConfigManager",
]
