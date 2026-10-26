"""Colour constants and application theme."""

from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication

try:
    import qdarktheme
    THEME_AVAILABLE = True
except ImportError:
    THEME_AVAILABLE = False


class Colors:
    GAZE = "#34c759"
    FIXATION = "#ff9500"
    TRAIL = "#007aff"
    SCANPATH = "#00d4ff"
    PLAYHEAD = "#ffffff"
    BG_DARK = "#000000"
    BG_PANEL = "#1a1a1a"
    BG_SIDEBAR = "#141414"
    BG_ANNOTATION = "#0f0f0f"
    BLINK = "#ff375f"
    SACCADE = "#bf5af2"
    EVENT = "#ffd60a"
    TEXT_DIM = "#8e8e93"
    ACCENT = "#0a84ff"
    BORDER = "#2a2a2a"


def apply_theme(app: QApplication):
    """Dark theme: pyqtdarktheme if installed, else a Fusion palette."""
    if THEME_AVAILABLE:
        app.setStyleSheet(qdarktheme.load_stylesheet("dark"))
        return
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor(Colors.BG_DARK))
    pal.setColor(QPalette.ColorRole.WindowText, QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.Base, QColor(Colors.BG_PANEL))
    pal.setColor(QPalette.ColorRole.Text, QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.Button, QColor("#3a3a3c"))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor("#ffffff"))
    app.setPalette(pal)
