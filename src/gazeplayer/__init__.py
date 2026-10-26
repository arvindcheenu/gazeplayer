"""GazePlayer: synchronised multi-participant eye-tracking video player with
AOI annotation, YOLO/FastSAM-assisted labelling and gaze-in-AOI metrics.

The top-level package is headless (no Qt import), so analysis scripts can do::

    from gazeplayer import load_session
    session, aois, tracker = load_session("recordings/song1/attempt1")
    metrics = tracker.compute_all_metrics()

The GUI lives in :mod:`gazeplayer.ui` and is started with ``gazeplayer``.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

from .aoi import AOI, AOIManager, BoundingBox
from .analysis import GazeAOIMetrics, GazeAOITracker, export_metrics_csv
from .config import PlayerConfig, find_config, load_config
from .session import ParticipantData, SessionData, SyncInfo, discover_participants

__version__ = "4.1.0"

__all__ = [
    "AOI", "AOIManager", "BoundingBox",
    "GazeAOIMetrics", "GazeAOITracker", "export_metrics_csv",
    "PlayerConfig", "find_config", "load_config",
    "ParticipantData", "SessionData", "SyncInfo", "discover_participants",
    "load_session", "__version__",
]


def load_session(session_dir: Union[str, Path],
                 config: Optional[Union[PlayerConfig, str, Path]] = None,
                 ) -> Tuple[SessionData, AOIManager, GazeAOITracker]:
    """Load a session, its AOIs and a gaze tracker in one call.

    *config* may be a :class:`PlayerConfig`, a path to a JSON config, or
    None to use the nearest ``gazeplayer.json`` (or defaults).
    """
    session_dir = Path(session_dir)
    if not isinstance(config, PlayerConfig):
        config = load_config(session_dir, Path(config) if config else None)
    session = SessionData(session_dir, config)
    aois = AOIManager(session_dir, config, list(session.participants))
    return session, aois, GazeAOITracker(aois, session, config)
