"""Study configuration.

Everything that used to be hard-coded for one particular study (participant
names, their colours, AOI label presets, the per-observer depth hierarchy …)
lives in a :class:`PlayerConfig`.  All fields are optional; with no config
file at all the player discovers participants from the session folder.

A config is a JSON file named ``gazeplayer.json``.  It is looked up in the
session folder and then in each parent folder, so one file at the root of a
``recordings/`` tree covers every session below it.  ``--config PATH`` on the
command line overrides the lookup.
"""

import json
import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

CONFIG_FILENAME = "gazeplayer.json"

# Colours handed out to participants that have no explicit colour.
DEFAULT_PARTICIPANT_PALETTE = [
    "#ff6b6b", "#4ecdc4", "#ffe66d", "#95e1d3",
    "#c792ea", "#f78c6c", "#82aaff", "#c3e88d",
    "#ffcb6b", "#89ddff", "#f07178", "#b2ccd6",
]

DEFAULT_EVENT_TYPES = [
    "musical_event", "gaze_event", "gaze_transition", "interaction", "annotation",
]


@dataclass
class PlayerConfig:
    """Study-specific settings.  See ``docs/configuration.md`` for details."""

    # Participant IDs in display order.  Each ID is matched (case-insensitively)
    # to a sub-folder of the session.  None = use every sub-folder with data.
    participants: Optional[List[str]] = None
    # participant ID -> hex colour.  Missing entries get a palette colour.
    participant_colors: Dict[str, str] = field(default_factory=dict)
    # Names offered when labelling a new AOI.  None = the participant IDs.
    aoi_presets: Optional[List[str]] = None
    # Labels bound to keys 1-9 in the Tracks tab.  None = participant IDs
    # followed by the AOI presets.
    quick_labels: Optional[List[str]] = None
    # observer ID -> AOI names, nearest first.  Breaks ties when gaze falls
    # on several overlapping AOIs.  Observers not listed use `participants`.
    depth_hierarchy: Dict[str, List[str]] = field(default_factory=dict)
    # Gaze within this many pixels of an AOI's edge still counts as on it.
    gaze_aoi_proximity_px: float = 40.0
    # Duration credited to each gaze sample when computing dwell time
    # (5 ms = 200 Hz, the Pupil Labs Neon / Invisible gaze rate).
    gaze_sample_ms: float = 5
    # Choices in the "Add Event" dialog (the field stays editable).
    event_types: List[str] = field(default_factory=lambda: list(DEFAULT_EVENT_TYPES))

    # Where this config was read from (not serialised).
    source: Optional[Path] = field(default=None, repr=False, compare=False)

    # ── Derived values ───────────────────────────────────────────────────

    def color_for(self, pid: str, index: int = 0) -> str:
        """Colour for participant *pid*; *index* picks a palette fallback."""
        if pid in self.participant_colors:
            return self.participant_colors[pid]
        return DEFAULT_PARTICIPANT_PALETTE[index % len(DEFAULT_PARTICIPANT_PALETTE)]

    def presets_for(self, participant_ids: List[str]) -> List[str]:
        if self.aoi_presets is not None:
            return list(self.aoi_presets)
        return list(participant_ids)

    def quick_labels_for(self, participant_ids: List[str]) -> List[str]:
        if self.quick_labels is not None:
            return list(self.quick_labels)[:9]
        labels = list(participant_ids)
        labels += [p for p in self.presets_for(participant_ids) if p not in labels]
        return labels[:9]

    def depth_order_for(self, observer: str, participant_ids: List[str]) -> List[str]:
        order = self.depth_hierarchy.get(observer)
        if order is None:
            order = self.participants or participant_ids
        return [name.lower() for name in order]

    # ── (De)serialisation ────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, data: dict) -> "PlayerConfig":
        known = {f.name for f in fields(cls)} - {"source"}
        unknown = set(data) - known
        for key in sorted(k for k in unknown if not k.startswith("_")):
            log.warning(f"⚠ Ignoring unknown config key '{key}'")
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("source")
        return d

    @classmethod
    def from_file(cls, path: Path) -> "PlayerConfig":
        path = Path(path)
        with open(path, "r") as f:
            cfg = cls.from_dict(json.load(f))
        cfg.source = path
        return cfg

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")


def find_config(session_dir: Path) -> Optional[Path]:
    """Return the nearest ``gazeplayer.json`` at or above *session_dir*."""
    session_dir = Path(session_dir).resolve()
    for folder in [session_dir, *session_dir.parents]:
        candidate = folder / CONFIG_FILENAME
        if candidate.is_file():
            return candidate
    return None


def load_config(session_dir: Optional[Path] = None,
                path: Optional[Path] = None) -> PlayerConfig:
    """Load an explicit config file, else the nearest one, else defaults."""
    if path is None and session_dir is not None:
        path = find_config(session_dir)
    if path is None:
        return PlayerConfig()
    cfg = PlayerConfig.from_file(path)
    log.info(f"✓ Config: {path}")
    return cfg
