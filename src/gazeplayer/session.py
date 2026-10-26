"""Session loading and cross-participant synchronisation.

A *session* is one folder with one sub-folder per participant::

    session/
        <participant>/  world.mp4, world.csv, gaze.csv, fixations.csv, ...
        session_aois.json      (written by the player)
        session_events.csv     (written by the player)

:class:`SessionData` loads every participant and aligns them on the
nanosecond timestamps in each ``world.csv``.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from .config import PlayerConfig

log = logging.getLogger(__name__)

# A participant folder must contain at least one of these.
PARTICIPANT_MARKER_FILES = ('gaze.csv', 'world.csv', 'world.mp4')


def _has_data(folder: Path) -> bool:
    return any((folder / f).exists() for f in PARTICIPANT_MARKER_FILES)


def discover_participants(session_dir: Path,
                          wanted: Optional[List[str]] = None) -> Dict[str, Path]:
    """Map participant IDs to their folders inside *session_dir*.

    With *wanted*, each ID is matched case-insensitively to a sub-folder and
    the result keeps the order of *wanted*.  If none of them match (or
    *wanted* is None) every sub-folder holding data becomes a participant,
    named after the folder and sorted alphabetically.
    """
    subdirs = sorted(x for x in Path(session_dir).iterdir()
                     if x.is_dir() and not x.name.startswith('.'))
    with_data = [d for d in subdirs if _has_data(d)]

    if wanted:
        matched: Dict[str, Path] = {}
        for pid in wanted:
            for d in with_data:
                if d.name.lower() == pid.lower():
                    matched[pid] = d
                    break
        if matched:
            return matched
        log.warning(f"  ⚠ No folders matching configured participants {wanted}")
        log.info("  → Auto-discovering participant folders...")

    return {d.name: d for d in with_data}

@dataclass
class SyncInfo:
    overlap_start_ns: int
    overlap_end_ns: int
    overlap_duration_s: float
    participant_offsets: Dict[str, float]
    participant_start_frames: Dict[str, int]


class ParticipantData:
    def __init__(self, data_dir: Path, participant_id: str):
        self.data_dir = data_dir
        self.participant_id = participant_id
        self.gaze: Optional[pd.DataFrame] = None
        self.fixations: Optional[pd.DataFrame] = None
        self.saccades: Optional[pd.DataFrame] = None
        self.world: Optional[pd.DataFrame] = None
        self.imu: Optional[pd.DataFrame] = None
        self.video_path: Optional[Path] = None
        self.video_cap: Optional[cv2.VideoCapture] = None
        self.start_timestamp: int = 0
        self.end_timestamp: int = 0
        self.fps: float = 30
        self.total_frames: int = 0
        self.frame_width: int = 1088
        self.frame_height: int = 1080
        # Frame reading state — avoids unreliable cv2 seeking
        self._current_video_pos: int = -1  # last frame index we read
        self._cached_frame: Optional[np.ndarray] = None
        self._cached_frame_idx: int = -1
        # YOLO + FastSAM data (lazy-loaded)
        self.yolo_data: Optional[dict] = None
        self.fastsam_data: Optional[dict] = None
        self._load_data()

    def _load_data(self):
        d = self.data_dir
        pid = self.participant_id

        # world.mp4 — load video first so we can fallback for timestamps
        vp = d / 'world.mp4'
        if vp.exists():
            self.video_path = vp
            self.video_cap = cv2.VideoCapture(str(vp))
            if self.video_cap.isOpened():
                self.fps = self.video_cap.get(cv2.CAP_PROP_FPS) or 30
                self.frame_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.frame_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                video_frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                # Quick verify: can we actually read a frame?
                ret, test_frame = self.video_cap.read()
                if ret:
                    # Cache this first frame so get_frame_at_timestamp can use it
                    self._cached_frame = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)
                    self._cached_frame_idx = 0
                    self._current_video_pos = 0
                    log.info(f"  ✓ {pid} video: {self.frame_width}x{self.frame_height} "
                          f"@ {self.fps:.1f}fps, {video_frame_count} frames (readable)")
                else:
                    log.warning(f"  ⚠ {pid} video: file opened but CANNOT READ frames!")
                    self.video_cap.release()
                    self.video_cap = None
            else:
                log.warning(f"  ⚠ {pid} video: OpenCV cannot open {vp.name}")
                self.video_cap = None
        else:
            log.warning(f"  ⚠ {pid}: no world.mp4 found")

        # world.csv — frame timestamps (required for sync)
        wp = d / 'world.csv'
        if wp.exists():
            self.world = pd.read_csv(wp)
            if 'timestamp [ns]' in self.world.columns:
                self.start_timestamp = int(self.world['timestamp [ns]'].min())
                self.end_timestamp = int(self.world['timestamp [ns]'].max())
                self.total_frames = len(self.world)
                dur_s = (self.end_timestamp - self.start_timestamp) / 1e9
                log.info(f"  ✓ {pid} world.csv: {self.total_frames} frames, "
                      f"duration={dur_s:.2f}s")
            else:
                log.warning(f"  ⚠ {pid} world.csv: missing 'timestamp [ns]' column! "
                      f"Columns: {list(self.world.columns)}")
                self.world = None
        else:
            log.warning(f"  ⚠ {pid}: no world.csv found")

        # Fallback: if we have video but no world.csv, generate synthetic timestamps
        if self.world is None and self.video_cap is not None:
            video_frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if video_frame_count > 0:
                log.info(f"  → Generating synthetic timestamps from video "
                      f"({video_frame_count} frames @ {self.fps:.1f}fps)")
                # Use arbitrary epoch-like base timestamp (2024-01-01 in nanoseconds)
                base_ns = 1704067200_000_000_000
                frame_dur_ns = int(1e9 / self.fps)
                timestamps = [base_ns + i * frame_dur_ns for i in range(video_frame_count)]
                self.world = pd.DataFrame({'timestamp [ns]': timestamps})
                self.start_timestamp = timestamps[0]
                self.end_timestamp = timestamps[-1]
                self.total_frames = video_frame_count
                dur_s = (self.end_timestamp - self.start_timestamp) / 1e9
                log.info(f"  ✓ {pid} synthetic timestamps: {self.total_frames} frames, "
                      f"duration={dur_s:.2f}s")

        # gaze.csv
        gp = d / 'gaze.csv'
        if gp.exists():
            self.gaze = pd.read_csv(gp)
            if self.start_timestamp and 'timestamp [ns]' in self.gaze.columns:
                self.gaze['timestamp_s'] = (self.gaze['timestamp [ns]'] - self.start_timestamp) / 1e9
            log.info(f"  ✓ {pid} gaze: {len(self.gaze)} samples")

        # fixations.csv
        fp = d / 'fixations.csv'
        if fp.exists():
            self.fixations = pd.read_csv(fp)
            if self.start_timestamp and 'start timestamp [ns]' in self.fixations.columns:
                self.fixations['start_s'] = (self.fixations['start timestamp [ns]'] - self.start_timestamp) / 1e9
                self.fixations['end_s'] = (self.fixations['end timestamp [ns]'] - self.start_timestamp) / 1e9
            log.info(f"  ✓ {pid} fixations: {len(self.fixations)}")

        # saccades.csv
        sp = d / 'saccades.csv'
        if sp.exists():
            self.saccades = pd.read_csv(sp)
            if self.start_timestamp and 'start timestamp [ns]' in self.saccades.columns:
                self.saccades['start_s'] = (self.saccades['start timestamp [ns]'] - self.start_timestamp) / 1e9
                self.saccades['end_s'] = (self.saccades['end timestamp [ns]'] - self.start_timestamp) / 1e9
            log.info(f"  ✓ {pid} saccades: {len(self.saccades)}")

        # imu.csv
        ip = d / 'imu.csv'
        if ip.exists():
            self.imu = pd.read_csv(ip)
            log.info(f"  ✓ {pid} imu: {len(self.imu)}")

        # YOLO detections (if previously computed)
        yp = d / 'yolo_detections.json'
        if yp.exists():
            try:
                with open(yp, 'r') as f:
                    self.yolo_data = json.load(f)
                n_tracks = len(self.yolo_data.get('tracks', {}))
                log.info(f"  ✓ {pid} YOLO: {n_tracks} tracks loaded")
            except Exception:
                pass

        # FastSAM masks (if previously computed)
        fsp = d / 'fastsam_masks.json'
        if fsp.exists():
            try:
                with open(fsp, 'r') as f:
                    self.fastsam_data = json.load(f)
                n_frames = len(self.fastsam_data.get('frames', {}))
                log.info(f"  ✓ {pid} FastSAM: {n_frames} frames loaded")
            except Exception:
                pass

    def get_frame_at_timestamp(self, ts_ns: int) -> Tuple[Optional[np.ndarray], int]:
        if self.video_cap is None or self.world is None:
            return None, 0
        idx = int(np.searchsorted(self.world['timestamp [ns]'].values, ts_ns))
        idx = max(0, min(idx, self.total_frames - 1))

        # Return cached frame if same index
        if idx == self._cached_frame_idx and self._cached_frame is not None:
            return self._cached_frame, idx

        # Sequential read: if next frame is within a small forward jump, just read ahead
        # This avoids the unreliable cv2.set(CAP_PROP_POS_FRAMES) on H.264/H.265
        if self._current_video_pos >= 0 and 0 < (idx - self._current_video_pos) <= 5:
            # Read and discard frames to advance sequentially
            for _ in range(idx - self._current_video_pos - 1):
                self.video_cap.grab()  # grab without decode = fast skip
            ret, frame = self.video_cap.read()
            if ret:
                self._current_video_pos = idx
                self._cached_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self._cached_frame_idx = idx
                return self._cached_frame, idx
            # Fall through to seek if sequential read failed

        # Hard seek — needed for jumps, backward seeks, or first read
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.video_cap.read()
        if ret:
            self._current_video_pos = idx
            self._cached_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._cached_frame_idx = idx
            return self._cached_frame, idx

        # If seek failed, try reopening the capture (OpenCV sometimes loses state)
        log.warning(f"  ⚠ {self.participant_id}: seek to frame {idx} failed, reopening video...")
        self.video_cap.release()
        self.video_cap = cv2.VideoCapture(str(self.video_path))
        if self.video_cap.isOpened():
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.video_cap.read()
            if ret:
                self._current_video_pos = idx
                self._cached_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self._cached_frame_idx = idx
                return self._cached_frame, idx

        return None, idx

    def get_gaze_at_timestamp(self, ts_ns: int, window_ms=33) -> Optional[Tuple[float, float]]:
        if self.gaze is None:
            return None
        w_ns = window_ms * 1e6
        mask = ((self.gaze['timestamp [ns]'] >= ts_ns - w_ns) &
                (self.gaze['timestamp [ns]'] <= ts_ns + w_ns))
        s = self.gaze[mask]
        if len(s) > 0:
            return (float(s['gaze x [px]'].mean()), float(s['gaze y [px]'].mean()))
        return None

    def get_fixations_at_timestamp(self, ts_ns: int, window_s=0.5):
        if self.fixations is None:
            return []
        t_s = (ts_ns - self.start_timestamp) / 1e9
        mask = (self.fixations['start_s'] >= t_s - window_s) & (self.fixations['start_s'] <= t_s)
        return [(r['fixation x [px]'], r['fixation y [px]'], r['duration [ms]'])
                for _, r in self.fixations[mask].iterrows()]

    def get_current_fixation(self, ts_ns: int):
        if self.fixations is None:
            return None
        t_s = (ts_ns - self.start_timestamp) / 1e9
        for _, r in self.fixations.iterrows():
            if r['start_s'] <= t_s <= r['end_s']:
                return (r['fixation x [px]'], r['fixation y [px]'], r['duration [ms]'])
        return None

    def close(self):
        if self.video_cap:
            self.video_cap.release()


class SessionData:
    def __init__(self, session_dir: Path, config: Optional[PlayerConfig] = None):
        self.session_dir = Path(session_dir)
        self.config = config or PlayerConfig()
        self.participants: Dict[str, ParticipantData] = {}
        self.sync_info: Optional[SyncInfo] = None
        self.events: Optional[pd.DataFrame] = None
        self.event_notes: dict = {}
        self.events_path = self.session_dir / "session_events.csv"
        self.notes_path = self.session_dir / "session_notes.json"
        self._load_participants()
        self._compute_sync()
        self._load_events()

    def _load_participants(self):
        log.info(f"\n📂 Loading session: {self.session_dir}")
        matched_known = discover_participants(self.session_dir, self.config.participants)
        if not matched_known:
            log.warning("  ⚠ No sub-folders contain gaze.csv, world.csv or world.mp4!")
            return

        # Load each matched participant
        for pid, subdir in matched_known.items():
            log.info(f"\n  Loading {pid} from {subdir.name}/...")
            files = sorted([f.name for f in subdir.iterdir() if f.is_file()])
            log.info(f"    Files: {files}")
            self.participants[pid] = ParticipantData(subdir, pid)

        if not self.participants:
            log.warning("⚠ No participants loaded!")

    def _compute_sync(self):
        if not self.participants:
            log.warning("⚠ No participants loaded — cannot compute sync")
            return

        # Include participants that have timestamps
        valid = [(pid, p) for pid, p in self.participants.items()
                 if p.start_timestamp > 0 and p.end_timestamp > 0 and p.world is not None]

        if not valid:
            # Last resort: any participant with video
            video_only = [(pid, p) for pid, p in self.participants.items()
                          if p.video_cap is not None]
            if video_only:
                log.warning("⚠ No participants with timestamps — generating from video fps")
                for pid, p in video_only:
                    fc = int(p.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    base_ns = 1704067200_000_000_000
                    frame_dur_ns = int(1e9 / p.fps)
                    timestamps = [base_ns + i * frame_dur_ns for i in range(max(1, fc))]
                    p.world = pd.DataFrame({'timestamp [ns]': timestamps})
                    p.start_timestamp = timestamps[0]
                    p.end_timestamp = timestamps[-1]
                    p.total_frames = len(timestamps)
                valid = video_only
            else:
                log.warning("⚠ No participants with valid timestamps or video")
                return

        log.info(f"\n🔄 Computing sync for {len(valid)} participants:")
        for pid, p in valid:
            dur_s = (p.end_timestamp - p.start_timestamp) / 1e9
            log.info(f"  {pid}: start={p.start_timestamp}, end={p.end_timestamp}, "
                  f"duration={dur_s:.2f}s, frames={p.total_frames}")

        if len(valid) == 1:
            pid, pd_ = valid[0]
            overlap_start = pd_.start_timestamp
            overlap_end = pd_.end_timestamp
            overlap_pids = {pid}
            log.info(f"  → Single participant mode: using full range")
        else:
            # Find the largest subset of participants that actually overlap temporally
            # Try all participants first, then progressively smaller subsets
            overlap_start, overlap_end, overlap_pids = self._find_best_overlap(valid)

        overlap_dur = (overlap_end - overlap_start) / 1e9
        if overlap_dur <= 0:
            log.warning(f"  ⚠ Computed duration is {overlap_dur:.4f}s, forcing to video length")
            longest_pid, longest_pd = max(valid, key=lambda x: x[1].total_frames)
            overlap_dur = longest_pd.total_frames / longest_pd.fps
            overlap_end = overlap_start + int(overlap_dur * 1e9)

        offsets = {}
        start_frames = {}
        for pid, pd_ in self.participants.items():
            if pd_.world is not None:
                ts_values = pd_.world['timestamp [ns]'].values
                if pid in overlap_pids:
                    offsets[pid] = (pd_.start_timestamp - overlap_start) / 1e9
                    idx = int(np.searchsorted(ts_values, overlap_start))
                    start_frames[pid] = max(0, min(idx, pd_.total_frames - 1))
                else:
                    # Non-overlapping participant: start from their frame 0
                    offsets[pid] = 0.0
                    start_frames[pid] = 0
                    log.warning(f"  ⚠ {pid}: timestamps don't overlap with main group — "
                          f"playing from their frame 0 (unsynchronized)")
            else:
                offsets[pid] = 0.0
                start_frames[pid] = 0

        self.sync_info = SyncInfo(overlap_start, overlap_end, overlap_dur, offsets, start_frames)
        log.info(f"\n✅ Sync: {overlap_dur:.2f}s across {len(self.participants)} participants")
        for pid in self.participants:
            synced = "synced" if pid in overlap_pids else "UNSYNCED"
            log.info(f"  {pid}: offset={offsets.get(pid, 0):.3f}s, "
                  f"start_frame={start_frames.get(pid, 0)} [{synced}]")

    @staticmethod
    def _find_best_overlap(valid):
        """Find the largest subset of participants with positive temporal overlap."""
        from itertools import combinations

        all_pids = [pid for pid, _ in valid]
        pid_data = {pid: p for pid, p in valid}

        # Try all participants first
        for subset_size in range(len(valid), 0, -1):
            for subset in combinations(all_pids, subset_size):
                starts = [pid_data[pid].start_timestamp for pid in subset]
                ends = [pid_data[pid].end_timestamp for pid in subset]
                o_start = max(starts)
                o_end = min(ends)
                if o_end > o_start:
                    dur = (o_end - o_start) / 1e9
                    if subset_size < len(valid):
                        excluded = set(all_pids) - set(subset)
                        log.warning(f"  ⚠ No full overlap! Best overlap: {list(subset)} "
                              f"({dur:.2f}s)")
                        log.info(f"    Excluded (no temporal overlap): {excluded}")
                    else:
                        log.info(f"  → All {len(valid)} participants overlap ({dur:.2f}s)")
                    return o_start, o_end, set(subset)

        # No overlap at all — use the longest recording
        longest_pid, longest_pd = max(valid,
            key=lambda x: x[1].end_timestamp - x[1].start_timestamp)
        log.warning(f"  ⚠ No overlap found at all! Using {longest_pid} as reference")
        return longest_pd.start_timestamp, longest_pd.end_timestamp, {longest_pid}

    def _load_events(self):
        if self.events_path.exists():
            self.events = pd.read_csv(self.events_path)
            for col in ['from_aoi', 'to_aoi', 'participant']:
                if col not in self.events.columns:
                    self.events[col] = ''
            log.info(f"✓ Loaded {len(self.events)} events")
        else:
            self.events = pd.DataFrame(columns=[
                'timestamp [ns]', 'name', 'type', 'participant', 'from_aoi', 'to_aoi'])
        if self.notes_path.exists():
            try:
                with open(self.notes_path, 'r') as f:
                    self.event_notes = {int(k): v for k, v in json.load(f).items()}
            except Exception:
                pass

    def get_current_timestamp(self, session_time_s: float) -> int:
        return self.sync_info.overlap_start_ns + int(session_time_s * 1e9)

    def add_event(self, data: dict):
        ts = data['timestamp [ns]']
        row = pd.DataFrame([{
            'timestamp [ns]': ts, 'name': data['name'], 'type': data['type'],
            'participant': data.get('participant', 'all'),
            'from_aoi': data.get('from_aoi', ''), 'to_aoi': data.get('to_aoi', '')
        }])
        self.events = pd.concat([self.events, row], ignore_index=True)
        if data.get('notes'):
            self.event_notes[int(ts)] = data['notes']
        self._save_events()

    def add_events_batch(self, events: List[dict]):
        rows = [{
            'timestamp [ns]': e['timestamp [ns]'], 'name': e['name'], 'type': e['type'],
            'participant': e.get('participant', 'all'),
            'from_aoi': e.get('from_aoi', ''), 'to_aoi': e.get('to_aoi', '')
        } for e in events]
        self.events = pd.concat([self.events, pd.DataFrame(rows)], ignore_index=True)
        self._save_events()

    def clear_transition_events(self):
        if self.events is not None:
            self.events = self.events[self.events['type'] != 'gaze_transition'].reset_index(drop=True)
            self._save_events()

    def _save_events(self):
        self.events.to_csv(self.events_path, index=False)
        with open(self.notes_path, 'w') as f:
            json.dump({str(k): v for k, v in self.event_notes.items()}, f, indent=2)

    def get_all_events_for_timeline(self) -> List[Tuple[float, str, str]]:
        if self.events is None or len(self.events) == 0:
            return []
        result = []
        for _, r in self.events.iterrows():
            t_s = (r['timestamp [ns]'] - self.sync_info.overlap_start_ns) / 1e9
            result.append((t_s, r['name'], r.get('participant', 'all')))
        return result

    def close(self):
        for p in self.participants.values():
            p.close()
