"""Areas of interest (AOIs).

An :class:`AOI` is a named bounding box that is keyframed independently in
every participant's world video.  :class:`AOIManager` owns all AOIs of a
session, persists them to ``session_aois.json`` and classifies gaze points.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .config import PlayerConfig

log = logging.getLogger(__name__)

AOI_COLORS = [
    "#ff5733", "#33ff57", "#3357ff", "#ff33f5",
    "#f5ff33", "#33fff5", "#ff8c33", "#8c33ff",
    "#33ff8c", "#ff3333", "#33ffcc", "#cc33ff",
]

@dataclass
class BoundingBox:
    x: float
    y: float
    width: float
    height: float

    def contains(self, px: float, py: float) -> bool:
        return (self.x <= px <= self.x + self.width and
                self.y <= py <= self.y + self.height)

    def is_in_bounds(self, frame_w: float, frame_h: float) -> bool:
        return not (self.x + self.width < 0 or self.x > frame_w or
                    self.y + self.height < 0 or self.y > frame_h)

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def distance_to(self, px: float, py: float) -> float:
        """Euclidean distance from *px, py* to the nearest edge.
        Returns 0 if the point is inside the box."""
        dx = max(self.x - px, 0.0, px - (self.x + self.width))
        dy = max(self.y - py, 0.0, py - (self.y + self.height))
        return (dx * dx + dy * dy) ** 0.5

    def intersection_area(self, other: 'BoundingBox') -> float:
        """Area of the intersection rectangle with *other*."""
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.width, other.x + other.width)
        y2 = min(self.y + self.height, other.y + other.height)
        if x1 < x2 and y1 < y2:
            return (x2 - x1) * (y2 - y1)
        return 0.0

    def overlap_fraction(self, other: 'BoundingBox') -> float:
        """Fraction of *this* bbox's area that is overlapped by *other*."""
        a = self.area
        if a <= 0:
            return 0.0
        return self.intersection_area(other) / a

    def to_dict(self) -> dict:
        return {'x': self.x, 'y': self.y, 'width': self.width, 'height': self.height}

    @classmethod
    def from_dict(cls, d: dict) -> 'BoundingBox':
        return cls(x=d['x'], y=d['y'], width=d['width'], height=d['height'])

    def copy(self) -> 'BoundingBox':
        return BoundingBox(self.x, self.y, self.width, self.height)


@dataclass
class AOI:
    """Area of Interest with per-participant keyframe animation."""
    name: str
    color: str
    participant_keyframes: Dict[str, Dict[int, BoundingBox]] = field(default_factory=dict)
    visible: bool = True
    # Frames that were synthetically pinned by update_aoi_isolated (not from YOLO/user draw).
    # These are swept away when a new edit is made, so interpolation flows to the next true keyframe.
    participant_anchor_frames: Dict[str, Set[int]] = field(default_factory=dict)
    # If True, only show bbox at frames with actual keyframes (for YOLO exact mode)
    exact_match: bool = False

    def get_bbox_at_frame(self, pid: str, frame_idx: int) -> Optional[BoundingBox]:
        keyframes = self.participant_keyframes.get(pid, {})
        if not keyframes:
            return None
        
        if self.exact_match:
            # Only return if there's a keyframe within 1 frame (for YOLO exact mode)
            for offset in [0, 1, -1]:
                check_frame = frame_idx + offset
                if check_frame in keyframes:
                    return keyframes[check_frame].copy()
            return None
        
        # Standard mode: use most recent keyframe
        valid = [kf for kf in keyframes if kf <= frame_idx]
        if not valid:
            return None
        prev_kf = max(valid)
        return keyframes[prev_kf].copy()

    def set_keyframe(self, pid: str, frame_idx: int, bbox: BoundingBox, is_anchor: bool = False):
        if pid not in self.participant_keyframes:
            self.participant_keyframes[pid] = {}
        if pid not in self.participant_anchor_frames:
            self.participant_anchor_frames[pid] = set()
        self.participant_keyframes[pid][frame_idx] = bbox
        if is_anchor:
            self.participant_anchor_frames[pid].add(frame_idx)
        else:
            # A true keyframe overwrites any previous anchor at this frame
            self.participant_anchor_frames[pid].discard(frame_idx)

    def is_anchor_frame(self, pid: str, frame_idx: int) -> bool:
        return frame_idx in self.participant_anchor_frames.get(pid, set())

    def remove_keyframe(self, pid: str, frame_idx: int):
        if pid in self.participant_keyframes and frame_idx in self.participant_keyframes[pid]:
            del self.participant_keyframes[pid][frame_idx]
        if pid in self.participant_anchor_frames:
            self.participant_anchor_frames[pid].discard(frame_idx)

    def get_keyframe_indices(self, pid: str) -> List[int]:
        return sorted(self.participant_keyframes.get(pid, {}).keys())

    def has_keyframe_at(self, pid: str, frame_idx: int) -> bool:
        return pid in self.participant_keyframes and frame_idx in self.participant_keyframes[pid]

    def to_dict(self) -> dict:
        return {
            'name': self.name, 'color': self.color, 'visible': self.visible,
            'exact_match': self.exact_match,
            'participant_keyframes': {
                pid: {str(k): v.to_dict() for k, v in kfs.items()}
                for pid, kfs in self.participant_keyframes.items()
            },
            'participant_anchor_frames': {
                pid: sorted(frames)
                for pid, frames in self.participant_anchor_frames.items()
                if frames
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'AOI':
        pkf = {}
        for pid, kfs in d.get('participant_keyframes', {}).items():
            pkf[pid] = {int(k): BoundingBox.from_dict(v) for k, v in kfs.items()}
        paf = {}
        for pid, frames in d.get('participant_anchor_frames', {}).items():
            paf[pid] = set(frames)
        return cls(name=d['name'], color=d['color'],
                   participant_keyframes=pkf, visible=d.get('visible', True),
                   participant_anchor_frames=paf,
                   exact_match=d.get('exact_match', False))


class AOIManager:
    def __init__(self, session_dir: Path, config: Optional[PlayerConfig] = None,
                 participant_ids: Optional[List[str]] = None):
        self.session_dir = session_dir
        self.config = config or PlayerConfig()
        self.participant_ids = list(participant_ids or [])
        self.aois: Dict[str, AOI] = {}
        self.aoi_path = session_dir / "session_aois.json"
        self._color_idx = 0
        self._load()

    def _load(self):
        if self.aoi_path.exists():
            try:
                with open(self.aoi_path, 'r') as f:
                    data = json.load(f)
                for ad in data.get('aois', []):
                    aoi = AOI.from_dict(ad)
                    self.aois[aoi.name] = aoi
                self._color_idx = len(self.aois)
                log.info(f"✓ Loaded {len(self.aois)} AOIs")
            except Exception as e:
                log.warning(f"⚠ Could not load AOIs: {e}")

    def save(self):
        data = {'aois': [a.to_dict() for a in self.aois.values()]}
        with open(self.aoi_path, 'w') as f:
            json.dump(data, f, indent=2)

    def create_aoi(self, name: str, participant_frames: Dict[str, int],
                   bbox: BoundingBox) -> AOI:
        if name in self.aois:
            raise ValueError(f"AOI '{name}' already exists")
        color = AOI_COLORS[self._color_idx % len(AOI_COLORS)]
        self._color_idx += 1
        aoi = AOI(name=name, color=color)
        for pid, fidx in participant_frames.items():
            aoi.set_keyframe(pid, fidx, bbox.copy())
        self.aois[name] = aoi
        self.save()
        return aoi

    def create_from_yolo_track(self, aoi_name: str, detections: dict,
                               track_id: int, pid: str, auto_save: bool = True) -> Optional[AOI]:
        """Create AOI from YOLO track data for one participant."""
        from .detection.yolo import detections_to_aoi_keyframes
        kfs = detections_to_aoi_keyframes(detections, track_id,
                                          detections.get('frame_count', 0))
        if not kfs:
            return None

        if aoi_name in self.aois:
            aoi = self.aois[aoi_name]
        else:
            color = AOI_COLORS[self._color_idx % len(AOI_COLORS)]
            self._color_idx += 1
            aoi = AOI(name=aoi_name, color=color, exact_match=True)
            self.aois[aoi_name] = aoi

        # Mark as exact_match for YOLO-imported AOIs
        aoi.exact_match = True

        for fidx, (x, y, w, h) in kfs.items():
            aoi.set_keyframe(pid, fidx, BoundingBox(x, y, w, h))

        if auto_save:
            self.save()
        return aoi

    def update_aoi(self, name: str, pid: str, frame_idx: int, bbox: BoundingBox):
        if name not in self.aois:
            raise ValueError(f"AOI '{name}' not found")
        self.aois[name].set_keyframe(pid, frame_idx, bbox)
        self.save()

    def update_aoi_isolated(self, name: str, pid: str, frame_idx: int, bbox: BoundingBox):
        """Update AOI at a frame.
        - Pins frame_idx-1 as an anchor to freeze past interpolation.
        - Sweeps any anchor keyframes between frame_idx+1 and the next TRUE keyframe
          (YOLO-sourced or user-drawn) so that interpolation flows naturally from the
          edited frame all the way to the next real detection."""
        if name not in self.aois:
            raise ValueError(f"AOI '{name}' not found")
        aoi = self.aois[name]

        # 1. Pin the previous frame as an anchor to freeze past interpolation.
        if not aoi.has_keyframe_at(pid, frame_idx):
            prev = frame_idx - 1
            if prev >= 0 and not aoi.has_keyframe_at(pid, prev):
                prev_bbox = aoi.get_bbox_at_frame(pid, prev)
                if prev_bbox:
                    aoi.set_keyframe(pid, prev, prev_bbox, is_anchor=True)

        # 2. Sweep anchor-only keyframes that sit between frame_idx and the next
        #    true (non-anchor) keyframe.  These are leftovers from previous edits and
        #    would otherwise block interpolation toward the real YOLO keyframe.
        keyframes = aoi.participant_keyframes.get(pid, {})
        for kf in sorted(kf for kf in keyframes if kf > frame_idx):
            if aoi.is_anchor_frame(pid, kf):
                aoi.remove_keyframe(pid, kf)   # remove stale anchor
            else:
                break  # hit a true keyframe — stop sweeping

        # 3. Set the edited frame as a true (non-anchor) keyframe.
        aoi.set_keyframe(pid, frame_idx, bbox, is_anchor=False)
        self.save()

    def delete_keyframe(self, name: str, pid: str, frame_idx: int):
        """Delete a single keyframe for one participant.
        Pins adjacent frames as anchors so the surrounding video appearance is
        preserved, but those pins can be swept away by a subsequent edit."""
        if name not in self.aois:
            raise ValueError(f"AOI '{name}' not found")
        aoi = self.aois[name]
        if not aoi.has_keyframe_at(pid, frame_idx):
            return
        # Pin both neighbours as anchors to preserve appearance across the gap.
        for adj in (frame_idx - 1, frame_idx + 1):
            if adj >= 0 and not aoi.has_keyframe_at(pid, adj):
                adj_bbox = aoi.get_bbox_at_frame(pid, adj)
                if adj_bbox:
                    aoi.set_keyframe(pid, adj, adj_bbox, is_anchor=True)
        aoi.remove_keyframe(pid, frame_idx)
        self.save()

    def delete_aoi(self, name: str):
        if name in self.aois:
            del self.aois[name]
            self.save()

    def rename_aoi(self, old: str, new: str):
        if old not in self.aois:
            raise ValueError(f"'{old}' not found")
        if new in self.aois:
            raise ValueError(f"'{new}' already exists")
        aoi = self.aois[old]
        aoi.name = new
        del self.aois[old]
        self.aois[new] = aoi
        self.save()

    def get_aoi_at_point(self, pid: str, fidx: int,
                         x: float, y: float, fw: float, fh: float) -> Optional[str]:
        """Legacy simple check — returns first AOI whose bbox contains the point."""
        for name, aoi in self.aois.items():
            if not aoi.visible:
                continue
            bbox = aoi.get_bbox_at_frame(pid, fidx)
            if bbox and bbox.is_in_bounds(fw, fh) and bbox.contains(x, y):
                return name
        return None

    def classify_gaze_aoi(self, pid: str, fidx: int,
                          gx: float, gy: float, fw: float, fh: float,
                          proximity_px: Optional[float] = None) -> str:
        """Classify a gaze point to an AOI using overlap + depth tiebreaking.

        Rules
        -----
        1. Collect every visible AOI whose bbox contains the gaze point *or*
           whose boundary is within *proximity_px* pixels of the point.
        2. No candidates  → ``'Other'``.
        3. ALL visible AOIs are candidates (and there are >1) → ``'Other'``.
        4. Exactly one candidate → return its name.
        5. Multiple candidates with pairwise overlap:
           a. For each candidate compute how much of its bbox is overlapped
              by the *other* candidates (sum of pairwise overlap fractions).
           b. The candidate with the **highest** overlap fraction wins
              (it is the most "covered" / specific region).
           c. If fractions are equal, use the observer's depth hierarchy
              as a tiebreaker (lower index wins).
           d. If still tied, prefer the candidate closest to the gaze point.
        """
        if proximity_px is None:
            proximity_px = self.config.gaze_aoi_proximity_px
        all_aois = self.get_all_aois_at_frame(pid, fidx)
        if not all_aois:
            return 'Other'

        # ── 1. find candidates ───────────────────────────────────────────
        candidates: Dict[str, Tuple[BoundingBox, float]] = {}   # name → (bbox, dist)
        for name, bbox in all_aois.items():
            if not bbox.is_in_bounds(fw, fh):
                continue
            dist = bbox.distance_to(gx, gy)
            if dist <= proximity_px:
                candidates[name] = (bbox, dist)

        # ── 2. no matches ────────────────────────────────────────────────
        if not candidates:
            return 'Other'

        # ── 3. all visible AOIs matched (>1) ─────────────────────────────
        if len(candidates) == len(all_aois) and len(all_aois) > 1:
            return 'Other'

        # ── 4. single match ──────────────────────────────────────────────
        if len(candidates) == 1:
            return next(iter(candidates))

        # ── 5. multiple matches — resolve by overlap then depth ──────────
        overlap_scores: Dict[str, float] = {}
        for name, (bbox, _) in candidates.items():
            total_frac = 0.0
            for other_name, (other_bbox, _) in candidates.items():
                if other_name != name:
                    total_frac += bbox.overlap_fraction(other_bbox)
            overlap_scores[name] = total_frac

        depth_order = self.config.depth_order_for(pid, self.participant_ids)

        def _sort_key(name: str):
            overlap = overlap_scores[name]
            # depth: position of the AOI name in the observer's hierarchy
            name_lower = name.lower()
            depth = (depth_order.index(name_lower)
                     if name_lower in depth_order
                     else len(depth_order))
            dist = candidates[name][1]
            # Higher overlap → wins  (negate for ascending sort)
            # Lower depth   → wins
            # Lower distance→ wins
            return (-overlap, depth, dist)

        return min(candidates, key=_sort_key)

    def get_all_aois_at_frame(self, pid: str, fidx: int) -> Dict[str, BoundingBox]:
        result = {}
        for name, aoi in self.aois.items():
            if not aoi.visible:
                continue
            bbox = aoi.get_bbox_at_frame(pid, fidx)
            if bbox:
                result[name] = bbox
        return result
