"""
Tests for IRH Desktop Application - Phase VI

These tests verify the desktop application components without
requiring PyQt6 (testing the non-GUI logic).

Author: Brandon D. McCrary
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock


# Add desktop src to path for testing
desktop_src = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(desktop_src))


class TestTransparencyEngine:
    """Tests for the Transparency Engine."""
    
    def test_message_level_enum(self):
        """Test MessageLevel enum values."""
        from irh_desktop.transparency.engine import MessageLevel
        
        assert MessageLevel.INFO.name == "INFO"
        assert MessageLevel.STEP.name == "STEP"
        assert MessageLevel.DETAIL.name == "DETAIL"
        assert MessageLevel.WHY.name == "WHY"
        assert MessageLevel.REF.name == "REF"
        assert MessageLevel.WARN.name == "WARN"
        assert MessageLevel.ERROR.name == "ERROR"
        assert MessageLevel.PASS.name == "PASS"
        assert MessageLevel.FAIL.name == "FAIL"
    
    def test_transparent_message_creation(self):
        """Test TransparentMessage creation."""
        from irh_desktop.transparency.engine import TransparentMessage, MessageLevel
        
        msg = TransparentMessage(
            level=MessageLevel.INFO,
            message="Test message",
            reference="§1.2.3, Eq. 1.13",
            equation="β_λ = -2λ̃ + (9/8π²)λ̃²"
        )
        
        assert msg.level == MessageLevel.INFO
        assert msg.message == "Test message"
        assert msg.reference == "§1.2.3, Eq. 1.13"
        assert msg.equation == "β_λ = -2λ̃ + (9/8π²)λ̃²"
        assert isinstance(msg.timestamp, datetime)
    
    def test_message_render_console(self):
        """Test console rendering of messages."""
        from irh_desktop.transparency.engine import TransparentMessage, MessageLevel
        
        msg = TransparentMessage(
            level=MessageLevel.INFO,
            message="Starting computation"
        )
        
        output = msg.render_console(use_color=False)
        assert "INFO" in output
        assert "Starting computation" in output
    
    def test_message_render_html(self):
        """Test HTML rendering of messages."""
        from irh_desktop.transparency.engine import TransparentMessage, MessageLevel
        
        msg = TransparentMessage(
            level=MessageLevel.STEP,
            message="Computing β_λ",
            reference="Eq. 1.13"
        )
        
        html = msg.render_html()
        assert "<div" in html
        assert "STEP" in html
        assert "Computing β_λ" in html
        assert "Eq. 1.13" in html
    
    def test_message_to_dict(self):
        """Test message serialization."""
        from irh_desktop.transparency.engine import TransparentMessage, MessageLevel
        
        msg = TransparentMessage(
            level=MessageLevel.DETAIL,
            message="Value computed",
            values={"λ̃": 52.637, "β_λ": 2.3e-14}
        )
        
        data = msg.to_dict()
        assert data["level"] == "DETAIL"
        assert data["message"] == "Value computed"
        assert data["values"]["λ̃"] == 52.637
    
    def test_transparency_engine_creation(self):
        """Test TransparencyEngine creation."""
        from irh_desktop.transparency.engine import TransparencyEngine
        
        engine = TransparencyEngine(verbosity=3)
        assert engine.verbosity == 3
        assert engine.show_equations is True
        assert engine.show_explanations is True
    
    def test_transparency_engine_info(self):
        """Test info message emission."""
        from irh_desktop.transparency.engine import TransparencyEngine, MessageLevel
        
        engine = TransparencyEngine(verbosity=5)
        messages = []
        engine.add_callback(lambda m: messages.append(m))
        
        engine.info("Starting RG flow", reference="§1.2")
        
        assert len(messages) == 1
        assert messages[0].level == MessageLevel.INFO
        assert messages[0].message == "Starting RG flow"
        assert messages[0].reference == "§1.2"
    
    def test_transparency_engine_step(self):
        """Test step message emission."""
        from irh_desktop.transparency.engine import TransparencyEngine
        
        engine = TransparencyEngine(verbosity=5)
        messages = []
        engine.add_callback(lambda m: messages.append(m))
        
        engine.step(
            "Computing β_λ",
            equation="β_λ = -2λ̃ + (9/8π²)λ̃²",
            reference="Eq. 1.13"
        )
        
        assert len(messages) == 1
        assert messages[0].equation == "β_λ = -2λ̃ + (9/8π²)λ̃²"
    
    def test_transparency_engine_detail(self):
        """Test detail message with values."""
        from irh_desktop.transparency.engine import TransparencyEngine
        
        engine = TransparencyEngine(verbosity=5)
        messages = []
        engine.add_callback(lambda m: messages.append(m))
        
        engine.detail("Result computed", values={"result": 2.3e-14})
        
        assert len(messages) == 1
        assert messages[0].values["result"] == 2.3e-14
    
    def test_transparency_engine_why(self):
        """Test explanation message."""
        from irh_desktop.transparency.engine import TransparencyEngine
        
        engine = TransparencyEngine(verbosity=5)
        messages = []
        engine.add_callback(lambda m: messages.append(m))
        
        engine.why("The β-function measures coupling running")
        
        assert len(messages) == 1
        assert "β-function" in messages[0].explanation
    
    def test_transparency_engine_passed(self):
        """Test pass message."""
        from irh_desktop.transparency.engine import TransparencyEngine, MessageLevel
        
        engine = TransparencyEngine(verbosity=5)
        messages = []
        engine.add_callback(lambda m: messages.append(m))
        
        engine.passed("Fixed point verified")
        
        assert len(messages) == 1
        assert messages[0].level == MessageLevel.PASS
        assert "✓" in messages[0].message
    
    def test_transparency_engine_verbosity_filter(self):
        """Test that verbosity filters messages."""
        from irh_desktop.transparency.engine import TransparencyEngine
        
        engine = TransparencyEngine(verbosity=2)  # Low verbosity
        messages = []
        engine.add_callback(lambda m: messages.append(m))
        
        engine.info("Info message")  # Should emit (level 2)
        engine.detail("Detail message")  # Should NOT emit (level 4)
        
        assert len(messages) == 1
        assert "Info" in messages[0].message
    
    def test_transparency_engine_context(self):
        """Test computation context tracking."""
        from irh_desktop.transparency.engine import TransparencyEngine
        
        engine = TransparencyEngine()
        
        engine.push_context("RG Flow")
        engine.push_context("Beta Functions")
        
        assert engine._context_stack == ["RG Flow", "Beta Functions"]
        
        popped = engine.pop_context()
        assert popped == "Beta Functions"
        assert engine._context_stack == ["RG Flow"]
    
    def test_transparency_engine_computation_lifecycle(self):
        """Test computation start/end lifecycle."""
        from irh_desktop.transparency.engine import TransparencyEngine
        
        engine = TransparencyEngine(verbosity=5)
        messages = []
        engine.add_callback(lambda m: messages.append(m))
        
        engine.computation_start("Fixed Point Verification", reference="§1.2.3")
        engine.computation_end(success=True)
        
        assert len(messages) >= 2
        assert "Starting" in messages[0].message
        assert "✓" in messages[-1].message


class TestConfigManager:
    """Tests for the Configuration Manager."""
    
    def test_config_manager_creation(self):
        """Test ConfigManager creation."""
        from irh_desktop.core.config_manager import ConfigManager
        
        config = ConfigManager()
        assert config.config is not None
    
    def test_default_config_values(self):
        """Test default configuration values."""
        from irh_desktop.core.config_manager import AppConfig
        
        config = AppConfig()
        assert config.version == "1.0"
        assert config.active_profile == "default"
        assert "default" in config.profiles
    
    def test_computation_profile_defaults(self):
        """Test ComputationProfile default values."""
        from irh_desktop.core.config_manager import ComputationProfile
        
        profile = ComputationProfile()
        assert profile.lattice_n_su2 == 50
        assert profile.lattice_n_u1 == 25
        assert profile.rg_method == "RK4"
        assert profile.precision_tolerance == 1e-12
    
    def test_config_get_set(self):
        """Test getting and setting config values."""
        from irh_desktop.core.config_manager import ConfigManager
        
        config = ConfigManager()
        
        # Test get
        assert config.get("appearance.dark_mode") is False
        
        # Test set
        result = config.set("appearance.dark_mode", True)
        assert result is True
        assert config.get("appearance.dark_mode") is True
    
    def test_config_create_profile(self):
        """Test creating a new profile."""
        from irh_desktop.core.config_manager import ConfigManager
        
        config = ConfigManager()
        
        profile = config.create_profile(
            name="test_profile",
            description="Test profile"
        )
        
        assert profile.name == "test_profile"
        assert "test_profile" in config.config.profiles
    
    def test_config_copy_profile(self):
        """Test copying an existing profile."""
        from irh_desktop.core.config_manager import ConfigManager
        
        config = ConfigManager()
        
        # Modify default
        config.config.profiles["default"].lattice_n_su2 = 100
        
        # Create copy
        new_profile = config.create_profile(
            name="copy",
            base_profile="default"
        )
        
        assert new_profile.lattice_n_su2 == 100
    
    def test_config_delete_profile(self):
        """Test deleting a profile."""
        from irh_desktop.core.config_manager import ConfigManager
        
        config = ConfigManager()
        config.create_profile("to_delete")
        
        result = config.delete_profile("to_delete")
        assert result is True
        assert "to_delete" not in config.config.profiles
    
    def test_cannot_delete_default_profile(self):
        """Test that default profile cannot be deleted."""
        from irh_desktop.core.config_manager import ConfigManager
        
        config = ConfigManager()
        result = config.delete_profile("default")
        
        assert result is False
        assert "default" in config.config.profiles
    
    def test_recent_files(self):
        """Test recent files tracking."""
        from irh_desktop.core.config_manager import ConfigManager
        
        config = ConfigManager()
        
        config.add_recent_file("/path/to/file1.yaml")
        config.add_recent_file("/path/to/file2.yaml")
        
        assert config.config.recent_files[0] == "/path/to/file2.yaml"
        assert config.config.recent_files[1] == "/path/to/file1.yaml"
    
    def test_recent_files_limit(self):
        """Test recent files list is limited to 20."""
        from irh_desktop.core.config_manager import ConfigManager
        
        config = ConfigManager()
        
        for i in range(25):
            config.add_recent_file(f"/path/to/file{i}.yaml")
        
        assert len(config.config.recent_files) == 20
    
    def test_config_reset_to_defaults(self):
        """Test resetting config to defaults."""
        from irh_desktop.core.config_manager import ConfigManager
        
        config = ConfigManager()
        config.set("appearance.dark_mode", True)
        config.reset_to_defaults()
        
        assert config.get("appearance.dark_mode") is False


class TestEngineManager:
    """Tests for the Engine Manager."""
    
    def test_engine_info_creation(self):
        """Test EngineInfo dataclass."""
        from irh_desktop.core.engine_manager import EngineInfo
        
        info = EngineInfo(
            path=Path("/opt/irh/engine"),
            version="21.0.0",
            commit="abc1234"
        )
        
        assert info.path == Path("/opt/irh/engine")
        assert info.version == "21.0.0"
        assert info.commit == "abc1234"
    
    def test_update_info_creation(self):
        """Test UpdateInfo dataclass."""
        from irh_desktop.core.engine_manager import UpdateInfo
        
        info = UpdateInfo(
            available=True,
            current_version="21.0.0",
            latest_version="21.0.1",
            changelog=["Bug fix", "New feature"]
        )
        
        assert info.available is True
        assert info.latest_version == "21.0.1"
        assert len(info.changelog) == 2
    
    def test_verification_result_creation(self):
        """Test VerificationResult dataclass."""
        from irh_desktop.core.engine_manager import VerificationResult
        
        result = VerificationResult(
            success=True,
            tests_passed=10,
            tests_failed=0
        )
        
        assert result.success is True
        assert result.tests_passed == 10
    
    def test_engine_manager_creation(self):
        """Test EngineManager creation."""
        from irh_desktop.core.engine_manager import EngineManager
        
        manager = EngineManager()
        assert manager.install_dir is not None
    
    def test_engine_manager_custom_dir(self):
        """Test EngineManager with custom directory."""
        from irh_desktop.core.engine_manager import EngineManager
        
        custom_path = Path("/tmp/irh_test")
        manager = EngineManager(install_dir=custom_path)
        
        assert manager.install_dir == custom_path
    
    def test_is_valid_engine_missing_files(self):
        """Test engine validation with missing files."""
        from irh_desktop.core.engine_manager import EngineManager
        import tempfile
        
        manager = EngineManager()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = manager._is_valid_engine(Path(tmpdir))
            assert result is False


class TestGlobalFunctions:
    """Tests for global transparency functions."""
    
    def test_get_transparency_engine(self):
        """Test getting global transparency engine."""
        from irh_desktop.transparency.engine import (
            get_transparency_engine, 
            TransparencyEngine
        )
        
        engine = get_transparency_engine()
        assert isinstance(engine, TransparencyEngine)
    
    def test_set_transparency_engine(self):
        """Test setting global transparency engine."""
        from irh_desktop.transparency.engine import (
            get_transparency_engine,
            set_transparency_engine,
            TransparencyEngine
        )
        
        custom_engine = TransparencyEngine(verbosity=5)
        set_transparency_engine(custom_engine)
        
        engine = get_transparency_engine()
        assert engine.verbosity == 5


# Tests for Phase VI verification
class TestPhaseVIVerification:
    """Verification tests for Phase VI implementation."""
    
    def test_all_core_modules_importable(self):
        """Test that all core modules can be imported."""
        from irh_desktop.core.engine_manager import EngineManager
        from irh_desktop.core.config_manager import ConfigManager
        from irh_desktop.transparency.engine import TransparencyEngine
        
        assert EngineManager is not None
        assert ConfigManager is not None
        assert TransparencyEngine is not None
    
    def test_transparency_engine_message_types(self):
        """Test all message types work correctly."""
        from irh_desktop.transparency.engine import TransparencyEngine
        
        engine = TransparencyEngine(verbosity=5)
        messages = []
        engine.add_callback(lambda m: messages.append(m))
        
        engine.info("Info message")
        engine.step("Step message")
        engine.detail("Detail message", values={"x": 1.0})
        engine.why("Explanation")
        engine.ref("§1.2.3", "Reference description")
        engine.warn("Warning")
        engine.error("Error")
        engine.passed("Pass")
        engine.failed("Fail")
        
        assert len(messages) == 9
    
    def test_config_profile_lifecycle(self):
        """Test complete profile lifecycle."""
        from irh_desktop.core.config_manager import ConfigManager
        
        config = ConfigManager()
        
        # Create
        profile = config.create_profile("lifecycle_test")
        assert "lifecycle_test" in config.config.profiles
        
        # Activate
        config.set_active_profile("lifecycle_test")
        assert config.config.active_profile == "lifecycle_test"
        
        # Get active
        active = config.get_active_profile()
        assert active.name == "lifecycle_test"
        
        # Delete
        config.delete_profile("lifecycle_test")
        assert "lifecycle_test" not in config.config.profiles
        assert config.config.active_profile == "default"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
