"""Gaze-in-AOI analysis: live scanpaths, dwell/transition metrics, transition
events, and CSV export.  Nothing here needs Qt, so it can run headless.
"""

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .aoi import AOIManager
from .config import PlayerConfig

if TYPE_CHECKING:
    from .session import SessionData

log = logging.getLogger(__name__)

@dataclass
class ScanpathPoint:
    x: float
    y: float
    timestamp_ns: int
    aoi_name: Optional[str]
    is_fixation: bool
    fixation_duration_ms: float = 0


@dataclass
class GazeAOIMetrics:
    dwell_time_ms: Dict[str, float] = field(default_factory=dict)
    fixation_count: Dict[str, int] = field(default_factory=dict)
    transitions: Dict[Tuple[str, str], int] = field(default_factory=dict)
    first_fixation_aoi: Optional[str] = None
    transition_sequence: List[Tuple[str, str]] = field(default_factory=list)


class GazeAOITracker:
    def __init__(self, aoi_manager: AOIManager, session: 'SessionData',
                 config: Optional[PlayerConfig] = None):
        self.aoi_manager = aoi_manager
        self.session = session
        self.config = config or aoi_manager.config
        self.current_aoi: Dict[str, Optional[str]] = {}
        self.scanpaths: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))

    def analyze_gaze_in_aoi(self, pid, ts_ns, gx, gy, fidx, fw, fh,
                            is_fix=False, fix_dur=0) -> Optional[str]:
        aoi_name = self.aoi_manager.classify_gaze_aoi(pid, fidx, gx, gy, fw, fh)
        if aoi_name == 'Other':
            aoi_name_display = 'Other'
        else:
            aoi_name_display = aoi_name
        pt = ScanpathPoint(gx, gy, ts_ns, aoi_name_display, is_fix, fix_dur)
        self.scanpaths[pid].append(pt)
        self.current_aoi[pid] = aoi_name_display
        return aoi_name_display

    def get_scanpath(self, pid: str) -> List[ScanpathPoint]:
        return list(self.scanpaths.get(pid, []))

    def clear_scanpaths(self):
        for sp in self.scanpaths.values():
            sp.clear()

    def generate_all_transition_events(self, progress_cb=None) -> List[dict]:
        all_events = []
        n_pids = max(1, len(self.session.participants))
        for pi, pid in enumerate(self.session.participants):
            pdata = self.session.participants.get(pid)
            if not pdata or pdata.gaze is None or pdata.world is None:
                continue
            if progress_cb:
                progress_cb(f"Processing {pid}...", pi * 100 // n_pids)
            frame_ts = pdata.world['timestamp [ns]'].values
            fw, fh = pdata.frame_width, pdata.frame_height
            prev_aoi = None
            total = len(pdata.gaze)
            for i, (_, row) in enumerate(pdata.gaze.iterrows()):
                if i % 1000 == 0 and progress_cb:
                    progress_cb(f"{pid}: {i}/{total}",
                                (pi * 100 + int(i / total * 100)) // n_pids)
                ts = row['timestamp [ns]']
                gx, gy = row['gaze x [px]'], row['gaze y [px]']
                fidx = int(np.searchsorted(frame_ts, ts))
                fidx = max(0, min(fidx, len(frame_ts) - 1))
                cur = self.aoi_manager.classify_gaze_aoi(pid, fidx, gx, gy, fw, fh)
                if cur == 'Other':
                    cur = None  # treat Other as unclassified for transition purposes
                if prev_aoi and cur and prev_aoi != cur:
                    all_events.append({
                        'timestamp [ns]': int(ts),
                        'name': f"gaze_{prev_aoi}_to_{cur}",
                        'type': 'gaze_transition',
                        'participant': pid,
                        'from_aoi': prev_aoi,
                        'to_aoi': cur,
                    })
                if cur:
                    prev_aoi = cur
        return all_events

    def compute_metrics(self, pid, start_f, end_f) -> GazeAOIMetrics:
        pdata = self.session.participants.get(pid)
        if not pdata or pdata.gaze is None or pdata.world is None:
            return GazeAOIMetrics()
        metrics = GazeAOIMetrics()
        frame_ts = pdata.world['timestamp [ns]'].values
        fw, fh = pdata.frame_width, pdata.frame_height
        s_ts = frame_ts[min(start_f, len(frame_ts)-1)]
        e_ts = frame_ts[min(end_f, len(frame_ts)-1)]
        mask = (pdata.gaze['timestamp [ns]'] >= s_ts) & (pdata.gaze['timestamp [ns]'] <= e_ts)
        gaze_subset = pdata.gaze[mask]
        
        if len(gaze_subset) == 0:
            return metrics
        
        # Vectorized: compute frame indices for all gaze points at once
        gaze_ts = gaze_subset['timestamp [ns]'].values
        gaze_x = gaze_subset['gaze x [px]'].values
        gaze_y = gaze_subset['gaze y [px]'].values
        fidx_all = np.searchsorted(frame_ts, gaze_ts)
        fidx_all = np.clip(fidx_all, 0, len(frame_ts) - 1)
        
        # Sample every Nth point to avoid processing millions of rows
        # Gaze is typically 200Hz, so step=10 gives ~20Hz which is plenty
        step = max(1, len(gaze_subset) // 10000)
        
        prev_aoi = None
        first = False
        for i in range(0, len(gaze_subset), step):
            fidx = int(fidx_all[i])
            gx, gy = float(gaze_x[i]), float(gaze_y[i])
            aoi = self.aoi_manager.classify_gaze_aoi(pid, fidx, gx, gy, fw, fh)
            # 'Other' is a valid dwell class but not used for transitions
            if aoi:
                if not first:
                    metrics.first_fixation_aoi = aoi
                    first = True
                # Scale dwell time by step size (gaze_sample_ms per sample)
                metrics.dwell_time_ms[aoi] = (metrics.dwell_time_ms.get(aoi, 0)
                                              + self.config.gaze_sample_ms * step)
            # Transitions: skip 'Other' so we only track named-AOI transitions
            aoi_for_trans = aoi if aoi != 'Other' else None
            if prev_aoi and aoi_for_trans and prev_aoi != aoi_for_trans:
                k = (prev_aoi, aoi_for_trans)
                metrics.transitions[k] = metrics.transitions.get(k, 0) + 1
                metrics.transition_sequence.append(k)
            if aoi_for_trans:
                prev_aoi = aoi_for_trans
        return metrics

    def compute_all_metrics(self) -> Dict[str, GazeAOIMetrics]:
        """Metrics over each participant's full recording."""
        metrics = {}
        for pid, pd_ in self.session.participants.items():
            if pd_.world is not None:
                metrics[pid] = self.compute_metrics(pid, 0, pd_.total_frames - 1)
        return metrics


# ── Export ───────────────────────────────────────────────────────────────────

METRICS_FILES = ('metrics_dwell_time.csv', 'metrics_transitions.csv', 'metrics_summary.csv')


def export_metrics_csv(metrics: Dict[str, GazeAOIMetrics], out_dir: Path) -> List[Path]:
    """Write dwell-time, transition and summary CSVs; return the paths written."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    written = []

    # Export dwell time CSV
    dwell_rows = []
    for pid, m in metrics.items():
        total_ms = sum(m.dwell_time_ms.values()) or 1
        for aoi, ms in m.dwell_time_ms.items():
            dwell_rows.append({
                'participant': pid,
                'aoi': aoi,
                'dwell_time_ms': ms,
                'dwell_percent': 100 * ms / total_ms,
            })
    if dwell_rows:
        dwell_file = out_path / METRICS_FILES[0]
        pd.DataFrame(dwell_rows).to_csv(dwell_file, index=False)
        written.append(dwell_file)

    # Export transitions CSV
    trans_rows = []
    for pid, m in metrics.items():
        for (from_aoi, to_aoi), count in m.transitions.items():
            trans_rows.append({
                'participant': pid,
                'from_aoi': from_aoi,
                'to_aoi': to_aoi,
                'count': count,
            })
    if trans_rows:
        trans_file = out_path / METRICS_FILES[1]
        pd.DataFrame(trans_rows).to_csv(trans_file, index=False)
        written.append(trans_file)

    # Export summary CSV
    summary_rows = []
    for pid, m in metrics.items():
        summary_rows.append({
            'participant': pid,
            'total_dwell_time_ms': sum(m.dwell_time_ms.values()),
            'num_aois_visited': len(m.dwell_time_ms),
            'total_transitions': sum(m.transitions.values()),
            'first_fixation_aoi': m.first_fixation_aoi or '',
        })
    if summary_rows:
        summary_file = out_path / METRICS_FILES[2]
        pd.DataFrame(summary_rows).to_csv(summary_file, index=False)
        written.append(summary_file)

    return written
