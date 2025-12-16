"""
IRH Desktop Application - Qt Application Setup

This module provides the QApplication setup and configuration for the
IRH Desktop application.

Theoretical Foundation:
    IRH21.md - Intrinsic Resonance Holography v21.0
    
Author: Brandon D. McCrary
"""

import sys
from typing import List, Optional
from pathlib import Path

# Try to import PyQt6, fall back gracefully if not available
try:
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt, QSettings
    from PyQt6.QtGui import QIcon, QPalette, QColor
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False
    QApplication = object  # Placeholder for type hints


class IRHDesktopApp:
    """
    IRH Desktop Application wrapper.
    
    Provides a high-level interface to the Qt application with
    IRH-specific configuration and theming.
    
    Attributes
    ----------
    app : QApplication
        The underlying Qt application
    settings : QSettings
        Application settings storage
        
    Examples
    --------
    >>> app = IRHDesktopApp(sys.argv)
    >>> app.setup_theme('dark')
    >>> sys.exit(app.run())
    """
    
    def __init__(self, argv: List[str]):
        """
        Initialize the IRH Desktop Application.
        
        Parameters
        ----------
        argv : List[str]
            Command-line arguments (typically sys.argv)
        """
        if not HAS_PYQT6:
            raise ImportError(
                "PyQt6 is required for IRH Desktop. "
                "Install with: pip install PyQt6"
            )
        
        # Set application metadata before creating QApplication
        QApplication.setApplicationName("IRH Desktop")
        QApplication.setApplicationVersion("21.0.0")
        QApplication.setOrganizationName("IRH Research")
        QApplication.setOrganizationDomain("irhresearch.org")
        
        # Create Qt application
        self.app = QApplication(argv)
        
        # Initialize settings
        self.settings = QSettings()
        
        # Setup default theme
        self._setup_default_theme()
        
        # Load application icon
        self._load_icon()
    
    def _setup_default_theme(self) -> None:
        """Configure the default application theme."""
        # Use fusion style for cross-platform consistency
        self.app.setStyle("Fusion")
        
        # Check for dark mode preference
        use_dark = self.settings.value("appearance/dark_mode", False, type=bool)
        if use_dark:
            self.setup_dark_theme()
    
    def _load_icon(self) -> None:
        """Load the application icon."""
        # Try to find icon in resources
        icon_paths = [
            Path(__file__).parent / "resources" / "icons" / "irh-desktop.png",
            Path("/opt/irh/desktop/share/icons/irh-desktop.png"),
            Path.home() / ".local/share/icons/irh-desktop.png",
        ]
        
        for icon_path in icon_paths:
            if icon_path.exists():
                self.app.setWindowIcon(QIcon(str(icon_path)))
                break
    
    def setup_dark_theme(self) -> None:
        """
        Apply dark theme to the application.
        
        Creates a professional dark color scheme suitable for
        scientific applications.
        """
        palette = QPalette()
        
        # Base colors
        dark_bg = QColor(30, 30, 30)
        dark_alt = QColor(45, 45, 45)
        dark_highlight = QColor(42, 130, 218)
        light_text = QColor(212, 212, 212)
        disabled_text = QColor(127, 127, 127)
        
        # Window
        palette.setColor(QPalette.ColorRole.Window, dark_bg)
        palette.setColor(QPalette.ColorRole.WindowText, light_text)
        
        # Base (text inputs)
        palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.ColorRole.AlternateBase, dark_alt)
        palette.setColor(QPalette.ColorRole.Text, light_text)
        
        # Buttons
        palette.setColor(QPalette.ColorRole.Button, dark_alt)
        palette.setColor(QPalette.ColorRole.ButtonText, light_text)
        
        # Highlights
        palette.setColor(QPalette.ColorRole.Highlight, dark_highlight)
        palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.white)
        
        # Disabled colors
        palette.setColor(QPalette.ColorGroup.Disabled, 
                        QPalette.ColorRole.WindowText, disabled_text)
        palette.setColor(QPalette.ColorGroup.Disabled, 
                        QPalette.ColorRole.Text, disabled_text)
        palette.setColor(QPalette.ColorGroup.Disabled, 
                        QPalette.ColorRole.ButtonText, disabled_text)
        
        # Links
        palette.setColor(QPalette.ColorRole.Link, dark_highlight)
        palette.setColor(QPalette.ColorRole.LinkVisited, QColor(128, 100, 200))
        
        # Tool tips
        palette.setColor(QPalette.ColorRole.ToolTipBase, dark_alt)
        palette.setColor(QPalette.ColorRole.ToolTipText, light_text)
        
        self.app.setPalette(palette)
        
        # Additional stylesheet for fine-tuning
        self.app.setStyleSheet("""
            QToolTip {
                border: 1px solid #3a3a3a;
                background-color: #2d2d2d;
                color: #d4d4d4;
                padding: 4px;
            }
            QGroupBox {
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                margin-top: 8px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QTabWidget::pane {
                border: 1px solid #3a3a3a;
                border-radius: 4px;
            }
            QScrollBar:vertical {
                border: none;
                background: #2d2d2d;
                width: 12px;
            }
            QScrollBar::handle:vertical {
                background: #5a5a5a;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #6a6a6a;
            }
        """)
    
    def setup_light_theme(self) -> None:
        """Apply light theme to the application."""
        # Reset to default palette
        self.app.setPalette(self.app.style().standardPalette())
        self.app.setStyleSheet("")
    
    def run(self) -> int:
        """
        Run the application event loop.
        
        Returns
        -------
        int
            Exit code from application
        """
        return self.app.exec()
    
    def quit(self) -> None:
        """Quit the application."""
        self.app.quit()


def create_app(argv: List[str]) -> 'QApplication':
    """
    Create and configure the Qt application.
    
    This is a convenience function that creates the QApplication
    with all IRH-specific configurations.
    
    Parameters
    ----------
    argv : List[str]
        Command-line arguments
        
    Returns
    -------
    QApplication
        Configured Qt application instance
    """
    if not HAS_PYQT6:
        raise ImportError(
            "PyQt6 is required for IRH Desktop. "
            "Install with: pip install PyQt6"
        )
    
    # Set application metadata
    QApplication.setApplicationName("IRH Desktop")
    QApplication.setApplicationVersion("21.0.0")
    QApplication.setOrganizationName("IRH Research")
    QApplication.setOrganizationDomain("irhresearch.org")
    
    # Create application
    app = QApplication(argv)
    app.setStyle("Fusion")
    
    return app
