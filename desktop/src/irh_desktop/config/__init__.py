"""
IRH Desktop - Configuration Package

Configuration management for the IRH Desktop application.

Author: Brandon D. McCrary
"""

from irh_desktop.core.config_manager import (
    ConfigManager,
    AppConfig,
    ComputationProfile,
    AppearanceSettings,
    EngineSettings,
)

__all__ = [
    "ConfigManager",
    "AppConfig",
    "ComputationProfile",
    "AppearanceSettings",
    "EngineSettings",
]
