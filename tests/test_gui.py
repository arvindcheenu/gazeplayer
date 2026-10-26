"""Offscreen smoke tests for the Qt UI."""

import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PyQt6.QtWidgets")


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_window_builds_one_tile_per_participant(app, session_with_aois):
    from gazeplayer.ui import MultiViewPlayer
    w = MultiViewPlayer(str(session_with_aois))
    assert list(w.participant_views) == ["Alice", "bob"]
    assert w.draw_buttons.count() == 2
    w._step(3)
    alice = w.participant_views["Alice"]
    assert alice.has_video
    assert alice.current_aoi == "left"
    w.close()


def test_configured_missing_participant_gets_placeholder_tile(app, session_dir):
    (session_dir / "gazeplayer.json").write_text(
        json.dumps({"participants": ["bob", "Alice", "carol"], "quick_labels": ["x", "y"]}))
    from gazeplayer.ui import MultiViewPlayer
    w = MultiViewPlayer(str(session_dir))
    assert list(w.participant_views) == ["bob", "Alice", "carol"]
    assert "no data" in w.participant_views["carol"].header.text()
    assert w.track_manager.quick_labels == ["x", "y"]
    w.close()


def test_yolo_import_dialog_returns_imports(app):
    from gazeplayer.ui.dialogs import YOLOImportDialog
    tracks = {"p": {"1": {"class": "person", "first_frame": 0, "last_frame": 40,
                          "suggested_aoi": "person_1"}}}
    data = {"p": {"fps": 30, "tracks": tracks["p"],
                  "frames": {str(i): [{"track_id": 1}] for i in range(25)}}}
    dlg = YOLOImportDialog(tracks, data, presets=["guitar"])
    assert dlg.get_imports() == [("p", 1, "person_1")]
