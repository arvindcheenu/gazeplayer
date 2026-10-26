"""PyQt6 user interface.  Importing this package requires PyQt6."""

import sys
from typing import Optional

from .main_window import MultiViewPlayer


def run_gui(session_path: Optional[str] = None, config_path: Optional[str] = None) -> int:
    """Open the player window (optionally on a session) and run the Qt loop."""
    from PyQt6.QtWidgets import QApplication
    from .theme import apply_theme

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("GazePlayer")
    apply_theme(app)
    window = MultiViewPlayer(session_path, config_path)
    window.show()
    return app.exec()


__all__ = ["MultiViewPlayer", "run_gui"]
