"""Synthetic two-participant session used by every test."""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

FPS = 30
N_FRAMES = 60
W, H = 160, 120
BASE_NS = 1_700_000_000_000_000_000


def _write_participant(folder: Path, start_ns: int, gaze_xy):
    folder.mkdir(parents=True)
    writer = cv2.VideoWriter(str(folder / "world.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(N_FRAMES):
        frame = np.full((H, W, 3), (i * 4) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    frame_ns = int(1e9 / FPS)
    world_ts = [start_ns + i * frame_ns for i in range(N_FRAMES)]
    pd.DataFrame({"timestamp [ns]": world_ts}).to_csv(folder / "world.csv", index=False)

    # 200 Hz gaze; first half on the left, second half on the right
    gaze_ns = np.arange(world_ts[0], world_ts[-1], 5_000_000)
    half = len(gaze_ns) // 2
    xs = np.where(np.arange(len(gaze_ns)) < half, gaze_xy[0][0], gaze_xy[1][0])
    ys = np.where(np.arange(len(gaze_ns)) < half, gaze_xy[0][1], gaze_xy[1][1])
    pd.DataFrame({"timestamp [ns]": gaze_ns, "gaze x [px]": xs,
                  "gaze y [px]": ys}).to_csv(folder / "gaze.csv", index=False)

    pd.DataFrame({
        "start timestamp [ns]": [world_ts[5]], "end timestamp [ns]": [world_ts[15]],
        "fixation x [px]": [gaze_xy[0][0]], "fixation y [px]": [gaze_xy[0][1]],
        "duration [ms]": [333.0],
    }).to_csv(folder / "fixations.csv", index=False)


@pytest.fixture
def session_dir(tmp_path) -> Path:
    """session/{Alice,bob}/... with Alice starting 0.5 s before bob."""
    root = tmp_path / "study" / "session1"
    _write_participant(root / "Alice", BASE_NS, [(20, 20), (140, 100)])
    _write_participant(root / "bob", BASE_NS + 500_000_000, [(20, 20), (140, 100)])
    (root / "notes").mkdir()  # a folder without data must be ignored
    return root


@pytest.fixture
def session_with_aois(session_dir) -> Path:
    """Adds two AOIs: 'left' (top-left) and 'right' (bottom-right)."""
    box = lambda x, y: {"x": x, "y": y, "width": 40, "height": 40}
    aois = {"aois": [
        {"name": "left", "color": "#ff0000",
         "participant_keyframes": {"Alice": {"0": box(0, 0)}, "bob": {"0": box(0, 0)}}},
        {"name": "right", "color": "#00ff00",
         "participant_keyframes": {"Alice": {"0": box(120, 80)}, "bob": {"0": box(120, 80)}}},
    ]}
    (session_dir / "session_aois.json").write_text(json.dumps(aois))
    return session_dir
