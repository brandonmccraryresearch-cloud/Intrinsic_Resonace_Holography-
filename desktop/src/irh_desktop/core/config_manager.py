"""
IRH Desktop - Configuration Manager

Manages application configuration including:
- User preferences and settings
- Computation profiles
- Theme settings
- Engine configuration

Configuration is stored in YAML format for human readability.

Author: Brandon D. McCrary
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
from datetime import datetime

import yaml

logger = logging.getLogger(__name__)

# Default configuration locations
SYSTEM_CONFIG = Path("/etc/irh/desktop.conf")
USER_CONFIG = Path.home() / ".config/irh/desktop.yaml"
DEFAULT_DATA_DIR = Path.home() / ".local/share/irh"


@dataclass
class ComputationProfile:
    """
    A saved computation profile with specific settings.
    
    Attributes
    ----------
    name : str
        Profile name
    description : str
        Profile description
    lattice_n_su2 : int
        SU(2) lattice size
    lattice_n_u1 : int
        U(1) lattice size
    lattice_spacing : float
        Lattice spacing parameter
    rg_method : str
        RG integration method
    rg_dt : float
        RG step size
    precision_dtype : str
        Numerical precision
    precision_tolerance : float
        Convergence tolerance
    """
    name: str = "Default"
    description: str = "Default computation settings"
    
    # Lattice settings
    lattice_n_su2: int = 50
    lattice_n_u1: int = 25
    lattice_spacing: float = 0.02
    
    # RG flow settings
    rg_method: str = "RK4"
    rg_dt: float = 0.001
    rg_t_uv: float = 10.0
    rg_t_ir: float = -20.0
    
    # Precision settings
    precision_dtype: str = "float64"
    precision_tolerance: float = 1e-12


@dataclass
class AppearanceSettings:
    """
    Application appearance settings.
    
    Attributes
    ----------
    dark_mode : bool
        Use dark theme
    font_size : int
        Base font size
    show_equations : bool
        Show LaTeX equations
    show_explanations : bool
        Show plain-language explanations
    verbosity : int
        Output verbosity level (1-5)
    """
    dark_mode: bool = False
    font_size: int = 12
    show_equations: bool = True
    show_explanations: bool = True
    verbosity: int = 3


@dataclass
class EngineSettings:
    """
    IRH engine settings.
    
    Attributes
    ----------
    engine_path : str
        Path to IRH engine
    auto_update : bool
        Enable automatic updates
    update_channel : str
        Update channel (stable/beta)
    """
    engine_path: str = ""
    auto_update: bool = True
    update_channel: str = "stable"


@dataclass
class AppConfig:
    """
    Complete application configuration.
    
    Attributes
    ----------
    version : str
        Configuration version
    appearance : AppearanceSettings
        Appearance settings
    engine : EngineSettings
        Engine settings
    profiles : Dict[str, ComputationProfile]
        Named computation profiles
    active_profile : str
        Currently active profile name
    recent_files : List[str]
        Recently opened files
    """
    version: str = "1.0"
    appearance: AppearanceSettings = field(default_factory=AppearanceSettings)
    engine: EngineSettings = field(default_factory=EngineSettings)
    profiles: Dict[str, ComputationProfile] = field(default_factory=lambda: {
        "default": ComputationProfile()
    })
    active_profile: str = "default"
    recent_files: List[str] = field(default_factory=list)
    
    def get_active_profile(self) -> ComputationProfile:
        """Get the currently active computation profile."""
        return self.profiles.get(self.active_profile, ComputationProfile())


class ConfigManager:
    """
    Manages IRH Desktop configuration.
    
    Handles loading, saving, and accessing configuration settings.
    Supports multiple configuration sources with priority:
    1. Command-line arguments (highest)
    2. User config file
    3. System config file
    4. Defaults (lowest)
    
    Parameters
    ----------
    config_path : Path, optional
        Custom configuration file path
        
    Examples
    --------
    >>> config = ConfigManager()
    >>> config.load()
    >>> profile = config.get_active_profile()
    >>> config.set("appearance.dark_mode", True)
    >>> config.save()
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the Configuration Manager.
        
        Parameters
        ----------
        config_path : Path, optional
            Custom configuration file path
        """
        self.config_path = config_path or USER_CONFIG
        self.config = AppConfig()
        self._loaded = False
    
    def load(self, path: Optional[str] = None) -> bool:
        """
        Load configuration from file.
        
        Parameters
        ----------
        path : str, optional
            Override path to load from
            
        Returns
        -------
        bool
            True if configuration was loaded
        """
        load_path = Path(path) if path else self.config_path
        
        if not load_path.exists():
            logger.info(f"Config file not found: {load_path}, using defaults")
            self._loaded = True
            return True
        
        try:
            with open(load_path, 'r') as f:
                data = yaml.safe_load(f)
            
            if data:
                self._load_from_dict(data)
            
            self._loaded = True
            logger.info(f"Configuration loaded from {load_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return False
    
    def _load_from_dict(self, data: Dict[str, Any]) -> None:
        """Load configuration from dictionary."""
        # Load appearance
        if "appearance" in data:
            app_data = data["appearance"]
            self.config.appearance = AppearanceSettings(
                dark_mode=app_data.get("dark_mode", False),
                font_size=app_data.get("font_size", 12),
                show_equations=app_data.get("show_equations", True),
                show_explanations=app_data.get("show_explanations", True),
                verbosity=app_data.get("verbosity", 3),
            )
        
        # Load engine settings
        if "engine" in data:
            eng_data = data["engine"]
            self.config.engine = EngineSettings(
                engine_path=eng_data.get("engine_path", ""),
                auto_update=eng_data.get("auto_update", True),
                update_channel=eng_data.get("update_channel", "stable"),
            )
        
        # Load profiles
        if "profiles" in data:
            self.config.profiles = {}
            for name, prof_data in data["profiles"].items():
                self.config.profiles[name] = ComputationProfile(
                    name=prof_data.get("name", name),
                    description=prof_data.get("description", ""),
                    lattice_n_su2=prof_data.get("lattice_n_su2", 50),
                    lattice_n_u1=prof_data.get("lattice_n_u1", 25),
                    lattice_spacing=prof_data.get("lattice_spacing", 0.02),
                    rg_method=prof_data.get("rg_method", "RK4"),
                    rg_dt=prof_data.get("rg_dt", 0.001),
                    rg_t_uv=prof_data.get("rg_t_uv", 10.0),
                    rg_t_ir=prof_data.get("rg_t_ir", -20.0),
                    precision_dtype=prof_data.get("precision_dtype", "float64"),
                    precision_tolerance=prof_data.get("precision_tolerance", 1e-12),
                )
        
        # Load other settings
        self.config.active_profile = data.get("active_profile", "default")
        self.config.recent_files = data.get("recent_files", [])
    
    def save(self, path: Optional[str] = None) -> bool:
        """
        Save configuration to file.
        
        Parameters
        ----------
        path : str, optional
            Override path to save to
            
        Returns
        -------
        bool
            True if configuration was saved
        """
        save_path = Path(path) if path else self.config_path
        
        try:
            # Ensure directory exists
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to dict
            data = {
                "version": self.config.version,
                "appearance": asdict(self.config.appearance),
                "engine": asdict(self.config.engine),
                "profiles": {
                    name: asdict(prof) 
                    for name, prof in self.config.profiles.items()
                },
                "active_profile": self.config.active_profile,
                "recent_files": self.config.recent_files[-20:],  # Keep last 20
            }
            
            # Write YAML
            with open(save_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Configuration saved to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Parameters
        ----------
        key : str
            Dot-separated key path (e.g., "appearance.dark_mode")
        default : Any
            Default value if key not found
            
        Returns
        -------
        Any
            Configuration value
        """
        parts = key.split(".")
        obj = self.config
        
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            elif isinstance(obj, dict) and part in obj:
                obj = obj[part]
            else:
                return default
        
        return obj
    
    def set(self, key: str, value: Any) -> bool:
        """
        Set a configuration value.
        
        Parameters
        ----------
        key : str
            Dot-separated key path
        value : Any
            Value to set
            
        Returns
        -------
        bool
            True if value was set
        """
        parts = key.split(".")
        
        if len(parts) == 1:
            if hasattr(self.config, parts[0]):
                setattr(self.config, parts[0], value)
                return True
            return False
        
        # Navigate to parent
        obj = self.config
        for part in parts[:-1]:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return False
        
        # Set final value
        if hasattr(obj, parts[-1]):
            setattr(obj, parts[-1], value)
            return True
        
        return False
    
    def get_active_profile(self) -> ComputationProfile:
        """
        Get the currently active computation profile.
        
        Returns
        -------
        ComputationProfile
            Active profile
        """
        return self.config.get_active_profile()
    
    def set_active_profile(self, name: str) -> bool:
        """
        Set the active computation profile.
        
        Parameters
        ----------
        name : str
            Profile name
            
        Returns
        -------
        bool
            True if profile was activated
        """
        if name in self.config.profiles:
            self.config.active_profile = name
            return True
        return False
    
    def create_profile(
        self,
        name: str,
        description: str = "",
        base_profile: Optional[str] = None
    ) -> ComputationProfile:
        """
        Create a new computation profile.
        
        Parameters
        ----------
        name : str
            Profile name
        description : str
            Profile description
        base_profile : str, optional
            Name of profile to copy settings from
            
        Returns
        -------
        ComputationProfile
            New profile
        """
        if base_profile and base_profile in self.config.profiles:
            # Copy from existing
            base = self.config.profiles[base_profile]
            profile = ComputationProfile(
                name=name,
                description=description or base.description,
                lattice_n_su2=base.lattice_n_su2,
                lattice_n_u1=base.lattice_n_u1,
                lattice_spacing=base.lattice_spacing,
                rg_method=base.rg_method,
                rg_dt=base.rg_dt,
                rg_t_uv=base.rg_t_uv,
                rg_t_ir=base.rg_t_ir,
                precision_dtype=base.precision_dtype,
                precision_tolerance=base.precision_tolerance,
            )
        else:
            profile = ComputationProfile(name=name, description=description)
        
        self.config.profiles[name] = profile
        return profile
    
    def delete_profile(self, name: str) -> bool:
        """
        Delete a computation profile.
        
        Parameters
        ----------
        name : str
            Profile name
            
        Returns
        -------
        bool
            True if profile was deleted
        """
        if name == "default":
            logger.warning("Cannot delete default profile")
            return False
        
        if name in self.config.profiles:
            del self.config.profiles[name]
            if self.config.active_profile == name:
                self.config.active_profile = "default"
            return True
        
        return False
    
    def add_recent_file(self, path: str) -> None:
        """
        Add a file to recent files list.
        
        Parameters
        ----------
        path : str
            File path to add
        """
        # Remove if already present
        if path in self.config.recent_files:
            self.config.recent_files.remove(path)
        
        # Add to front
        self.config.recent_files.insert(0, path)
        
        # Keep only last 20
        self.config.recent_files = self.config.recent_files[:20]
    
    def get_data_dir(self) -> Path:
        """
        Get the data directory for results and exports.
        
        Returns
        -------
        Path
            Data directory
        """
        data_dir = DEFAULT_DATA_DIR
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir
    
    def reset_to_defaults(self) -> None:
        """Reset all settings to defaults."""
        self.config = AppConfig()
        logger.info("Configuration reset to defaults")
