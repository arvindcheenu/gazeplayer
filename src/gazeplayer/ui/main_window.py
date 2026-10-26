"""Main window: synchronised participant grid, controls, timeline, sidebar."""

from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QDialog, QMessageBox, QStatusBar, QGridLayout, QInputDialog,
    QTabWidget, QProgressDialog, QFileDialog, QLayout,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QShortcut, QKeySequence

from ..aoi import AOIManager, BoundingBox
from ..analysis import GazeAOITracker, METRICS_FILES, export_metrics_csv
from ..config import PlayerConfig, load_config
from ..detection.yolo import get_detection_at_frame
from ..session import SessionData
from .aoi_panel import AOIPanel
from .dialogs import AddEventDialog, MetricsDialog, YOLOImportDialog
from .participant_view import ParticipantView
from .theme import Colors
from .timeline import SyncTimeline
from .track_manager import YOLOTrackManager

APP_TITLE = "👁 GazePlayer"


class MultiViewPlayer(QMainWindow):
    def __init__(self, session_path: Optional[str] = None,
                 config_path: Optional[str] = None):
        super().__init__()
        self.setWindowTitle(f"{APP_TITLE} — Multi-View + YOLO + FastSAM")
        self.setMinimumSize(1400, 900); self.resize(1600, 1000)

        self.session: Optional[SessionData] = None
        self.aoi_manager: Optional[AOIManager] = None
        self.gaze_tracker: Optional[GazeAOITracker] = None
        self.current_time_s = 0.0
        self.is_playing = False
        self.participant_views: Dict[str, ParticipantView] = {}
        self.drawing_participant: Optional[str] = None
        self.config_path = Path(config_path) if config_path else None
        self.config = PlayerConfig()

        self._setup_ui()
        self._setup_shortcuts()

        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self._advance_frame)

        if session_path:
            self._load_session(Path(session_path))

    def _setup_shortcuts(self):
        QShortcut(QKeySequence("Space"), self, self._toggle_play)
        QShortcut(QKeySequence("Left"), self, lambda: self._step(-1))
        QShortcut(QKeySequence("Right"), self, lambda: self._step(1))
        QShortcut(QKeySequence("Home"), self, lambda: self._seek_time(0))
        QShortcut(QKeySequence("End"), self, self._seek_end)
        QShortcut(QKeySequence("E"), self, self._add_event)
        # Keys 1-6 reserved for track manager quick rename
        QShortcut(QKeySequence("G"), self, lambda: self.chk_gaze.setChecked(not self.chk_gaze.isChecked()))
        QShortcut(QKeySequence("F"), self, lambda: self.chk_fix.setChecked(not self.chk_fix.isChecked()))
        QShortcut(QKeySequence("T"), self, lambda: self.chk_trail.setChecked(not self.chk_trail.isChecked()))
        QShortcut(QKeySequence("S"), self, lambda: self.chk_scan.setChecked(not self.chk_scan.isChecked()))
        QShortcut(QKeySequence("B"), self, lambda: self.chk_aois.setChecked(not self.chk_aois.isChecked()))
        QShortcut(QKeySequence("Y"), self, lambda: self.chk_yolo.setChecked(not self.chk_yolo.isChecked()))
        QShortcut(QKeySequence("M"), self, self._open_track_manager)
        QShortcut(QKeySequence("Ctrl+O"), self, self._open_session_dialog)
        QShortcut(QKeySequence("Escape"), self, self._cancel_draw)
        QShortcut(QKeySequence("Delete"), self, self._delete_selected_keyframe)
        # Backspace reserved for track manager deletion

    def _setup_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8); main_layout.setSpacing(8)

        # Left: videos + controls + timeline
        left = QWidget(); ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0); ll.setSpacing(8)

        # Video grid (filled by _build_participant_views once a session loads)
        grid_w = QWidget(); self.grid = QGridLayout(grid_w)
        self.grid.setSpacing(4); self.grid.setContentsMargins(0, 0, 0, 0)
        self.placeholder = QLabel("No session loaded — press Ctrl+O or click 📂 Open")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setStyleSheet(f"color:{Colors.TEXT_DIM};font-size:14px;")
        self.grid.addWidget(self.placeholder, 0, 0)
        ll.addWidget(grid_w, 1)

        # Control bar
        ctrl = QWidget()
        ctrl.setStyleSheet(f"background-color:{Colors.BG_PANEL};border-radius:8px;")
        cl = QHBoxLayout(ctrl); cl.setContentsMargins(16, 8, 16, 8); cl.setSpacing(12)

        self.btn_open = QPushButton("📂 Open")
        self.btn_open.setToolTip("Open a session folder (Ctrl+O)")
        self.btn_open.setStyleSheet("QPushButton{color:white;font-size:12px;padding:6px 10px;"
                                    "border:none;border-radius:6px;"
                                    "background-color:rgba(255,255,255,0.1);}"
                                    "QPushButton:hover{background-color:rgba(255,255,255,0.2);}")
        self.btn_open.clicked.connect(self._open_session_dialog); cl.addWidget(self.btn_open)

        self.btn_play = QPushButton("▶")
        self.btn_play.setFixedSize(44, 44)
        self.btn_play.setStyleSheet(f"QPushButton{{background-color:{Colors.ACCENT};border-radius:22px;"
                                    f"font-size:18px;color:white;border:none;}}")
        self.btn_play.clicked.connect(self._toggle_play); cl.addWidget(self.btn_play)

        self.time_label = QLabel("0:00.00 / 0:00.00")
        self.time_label.setStyleSheet("color:white;font-size:14px;font-weight:500;"
                                      "font-family:'SF Mono','Menlo',monospace;min-width:140px;")
        cl.addWidget(self.time_label); cl.addStretch()

        # Draw buttons (one per participant, added on session load)
        cl.addWidget(QLabel("Draw AOI:"))
        self.draw_buttons = QHBoxLayout(); self.draw_buttons.setSpacing(4)
        cl.addLayout(self.draw_buttons)
        cl.addSpacing(16)

        def mk_tog(label, color, checked=True):
            b = QPushButton(label); b.setCheckable(True); b.setChecked(checked)
            b.setStyleSheet(f"QPushButton{{color:{color};font-size:10px;font-weight:bold;"
                            f"padding:4px 8px;border-radius:10px;border:none;"
                            f"background-color:rgba(255,255,255,0.1);}}"
                            f"QPushButton:!checked{{background-color:transparent;color:#555;}}")
            return b

        self.chk_gaze = mk_tog("Gaze", Colors.GAZE); self.chk_gaze.toggled.connect(self._update_opts); cl.addWidget(self.chk_gaze)
        self.chk_fix = mk_tog("Fix", Colors.FIXATION); self.chk_fix.toggled.connect(self._update_opts); cl.addWidget(self.chk_fix)
        self.chk_trail = mk_tog("Trail", Colors.TRAIL); self.chk_trail.toggled.connect(self._update_opts); cl.addWidget(self.chk_trail)
        self.chk_scan = mk_tog("Scan", Colors.SCANPATH, False); self.chk_scan.toggled.connect(self._update_opts); cl.addWidget(self.chk_scan)
        self.chk_aois = mk_tog("AOIs", "#00ff00"); self.chk_aois.toggled.connect(self._update_opts); cl.addWidget(self.chk_aois)
        self.chk_yolo = mk_tog("YOLO", "#00ff88", False); self.chk_yolo.toggled.connect(self._update_opts); cl.addWidget(self.chk_yolo)

        ll.addWidget(ctrl)

        self.timeline = SyncTimeline(); self.timeline.seek_requested.connect(self._seek_time)
        ll.addWidget(self.timeline)
        main_layout.addWidget(left, 1)

        # Right panel: tabbed sidebar (AOIs | Tracks)
        right = QWidget(); right.setFixedWidth(500); rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(4)

        self.sidebar_tabs = QTabWidget()
        self.sidebar_tabs.setStyleSheet(
            f"QTabWidget::pane{{border:1px solid {Colors.BORDER};"
            f"background-color:{Colors.BG_PANEL};}}"
            f"QTabBar::tab{{background:{Colors.BG_ANNOTATION};color:#aaa;"
            f"padding:6px 12px;border:1px solid {Colors.BORDER};font-size:11px;"
            f"font-weight:bold;}}"
            f"QTabBar::tab:selected{{background:{Colors.BG_PANEL};color:white;}}")

        # Tab 1: AOIs
        aoi_tab = QWidget()
        aoi_layout = QVBoxLayout(aoi_tab)
        aoi_layout.setContentsMargins(0, 0, 0, 0); aoi_layout.setSpacing(4)
        self.aoi_panel = AOIPanel()
        self.aoi_panel.create_aoi_clicked.connect(self._start_draw_first)
        self.aoi_panel.import_yolo_clicked.connect(self._import_yolo)
        self.aoi_panel.manage_tracks_clicked.connect(self._open_track_manager)
        self.aoi_panel.fastsam_clicked.connect(self._toggle_fastsam_mode)
        self.aoi_panel.aoi_selected.connect(self._on_aoi_selected)
        self.aoi_panel.aoi_deleted.connect(self._on_aoi_deleted)
        self.aoi_panel.aoi_renamed.connect(self._on_aoi_renamed)
        self.aoi_panel.delete_all_aois_clicked.connect(self._on_delete_all_aois)
        self.aoi_panel.analyze_clicked.connect(self._analyze)
        self.aoi_panel.export_metrics_clicked.connect(self._export_metrics)
        self.aoi_panel.gen_events_clicked.connect(self._gen_transitions)
        self.aoi_panel.clear_events_clicked.connect(self._clear_transitions)
        aoi_layout.addWidget(self.aoi_panel)

        evt_btn = QPushButton("+ Add Event (E)")
        evt_btn.clicked.connect(self._add_event)
        evt_btn.setStyleSheet(f"QPushButton{{background-color:#2d2d2d;color:white;"
                              f"border:1px solid {Colors.BORDER};padding:8px;border-radius:4px;}}"
                              f"QPushButton:hover{{background-color:#3d3d3d;}}")
        aoi_layout.addWidget(evt_btn)
        aoi_layout.addStretch()
        self.sidebar_tabs.addTab(aoi_tab, "📦 AOIs")

        # Tab 2: YOLO Track Manager
        self.track_manager = YOLOTrackManager()
        self.track_manager.tracks_changed.connect(self._on_tracks_changed)
        self.track_manager.highlight_track.connect(self._on_highlight_track)
        self.track_manager.seek_to_frame.connect(self._on_seek_to_frame)
        self.track_manager.import_requested.connect(self._on_import_tracks)
        self.sidebar_tabs.addTab(self.track_manager, "🔧 Tracks")

        rl.addWidget(self.sidebar_tabs)
        main_layout.addWidget(right)

        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet(f"QStatusBar{{background-color:{Colors.BG_DARK};"
                                      f"color:{Colors.TEXT_DIM};font-size:11px;}}")
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready • Ctrl+O: Open session • E: Event • Y: YOLO overlay • "
                                    "M: Manage tracks • Space: Play")

    def _clear_layout(self, layout: QLayout):
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _build_participant_views(self, pids: List[str]):
        """(Re)create one video tile and one draw button per participant."""
        self._clear_layout(self.grid)
        self._clear_layout(self.draw_buttons)
        self.placeholder = None
        self.participant_views = {}
        presets = self.config.presets_for(pids)
        cols = max(1, int(np.ceil(np.sqrt(len(pids)))))
        for i, pid in enumerate(pids):
            color = self.config.color_for(pid, i)
            view = ParticipantView(pid, color=color, aoi_presets=presets)
            view.aoi_created.connect(self._on_aoi_created)
            view.aoi_updated.connect(self._on_aoi_updated)
            view.aoi_selected.connect(self._on_aoi_selected)
            view.aoi_keyframe_deleted.connect(self._on_aoi_keyframe_deleted)
            view.fastsam_click.connect(self._on_fastsam_click)
            self.participant_views[pid] = view
            self.grid.addWidget(view, i // cols, i % cols)

            b = QPushButton(str(i + 1)); b.setFixedSize(28, 28)
            b.setToolTip(f"Draw AOI on {pid}")
            b.setStyleSheet(f"QPushButton{{background-color:{color};"
                            f"color:black;border:none;font-weight:bold;border-radius:4px;}}")
            b.clicked.connect(lambda _, p=pid: self._start_draw(p))
            self.draw_buttons.addWidget(b)
        self.timeline.set_participant_colors(
            {pid: self.config.color_for(pid, i) for i, pid in enumerate(pids)})
        self._apply_view_opts()

    def _open_session_dialog(self):
        start = str(self.session.session_dir) if self.session else str(Path.home())
        path = QFileDialog.getExistingDirectory(self, "Open Session Folder", start)
        if path:
            self._load_session(Path(path))

    # ── Session Loading ──────────────────────────────────────────────────────

    def _load_session(self, path: Path):
        if not path.exists():
            self.status_bar.showMessage(f"Error: {path} not found"); return
        if not path.is_dir():
            path = path.parent

        # Reset state
        self.current_time_s = 0.0
        self._pause()
        if self.session:
            self.session.close()

        self.config = load_config(path, self.config_path)
        self.session = SessionData(path, self.config)

        if not self.session.participants:
            self.status_bar.showMessage("Error: No participant data found!")
            QMessageBox.warning(self, "No Data",
                f"No participant data found in:\n{path}\n\n"
                f"Expected subfolders with world.mp4 and/or world.csv.\n"
                f"Folders found: {[x.name for x in path.iterdir() if x.is_dir()]}")
            return

        # One tile per participant; configured-but-missing ones show "no data"
        loaded_pids = list(self.session.participants.keys())
        view_pids = loaded_pids + [p for p in (self.config.participants or [])
                                   if p not in self.session.participants]
        self._build_participant_views(view_pids)
        for pid, view in self.participant_views.items():
            if pid in self.session.participants:
                pd_ = self.session.participants[pid]
                label = f"{pid.upper()}"
                if pd_.video_cap:
                    label += f" ({pd_.frame_width}x{pd_.frame_height})"
                view.header.setText(label)
                view.header.setVisible(True)
                view.setVisible(True)
            else:
                view.header.setText(f"{pid.upper()} (no data)")
                # Don't hide — keep the grid layout stable

        self.aoi_manager = AOIManager(path, self.config, loaded_pids)
        self.gaze_tracker = GazeAOITracker(self.aoi_manager, self.session, self.config)
        self.track_manager.set_quick_labels(self.config.quick_labels_for(loaded_pids))
        self.track_manager.set_session(self.session)

        if self.session.sync_info:
            self.timeline.set_duration(self.session.sync_info.overlap_duration_s)
            self.timeline.set_events(self.session.get_all_events_for_timeline())
            self._update_aoi_panel()
            self._display_frame()
            dur = self.session.sync_info.overlap_duration_s
            np_ = len(self.session.participants)
            na = len(self.aoi_manager.aois)
            self.setWindowTitle(f"{APP_TITLE} — {path.name}")
            self.status_bar.showMessage(
                f"📂 {path.name} • {np_} participants • {dur:.1f}s • {na} AOIs • "
                f"E: Event • Y: YOLO • M: Manage tracks")
        else:
            self.status_bar.showMessage("Error: Could not compute sync — check terminal output")

    # ── Display ──────────────────────────────────────────────────────────────

    def _read_frame_by_index(self, pd_, fidx):
        """Read a frame by index directly, bypassing timestamp lookup."""
        if pd_.video_cap is None:
            return None
        if fidx == pd_._cached_frame_idx and pd_._cached_frame is not None:
            return pd_._cached_frame
        if pd_._current_video_pos >= 0 and 0 < (fidx - pd_._current_video_pos) <= 5:
            for _ in range(fidx - pd_._current_video_pos - 1):
                pd_.video_cap.grab()
            ret, frame = pd_.video_cap.read()
            if ret:
                pd_._current_video_pos = fidx
                pd_._cached_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pd_._cached_frame_idx = fidx
                return pd_._cached_frame
        pd_.video_cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = pd_.video_cap.read()
        if ret:
            pd_._current_video_pos = fidx
            pd_._cached_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pd_._cached_frame_idx = fidx
            return pd_._cached_frame
        return None

    def _display_frame(self):
        if not self.session or not self.session.sync_info:
            return
        ts = self.session.get_current_timestamp(self.current_time_s)
        for pid, pd_ in self.session.participants.items():
            view = self.participant_views.get(pid)
            if not view: continue

            # Check if this participant's data overlaps with the sync window
            if pd_.world is not None:
                ts_values = pd_.world['timestamp [ns]'].values
                if ts_values[0] > self.session.sync_info.overlap_end_ns or \
                   ts_values[-1] < self.session.sync_info.overlap_start_ns:
                    # Non-overlapping participant: map current_time_s to their frames directly
                    fidx = int(self.current_time_s * pd_.fps)
                    fidx = max(0, min(fidx, pd_.total_frames - 1))
                    frame = self._read_frame_by_index(pd_, fidx)
                    view.set_frame(frame, fidx)
                    # Use their own timestamp for gaze lookup
                    own_ts = int(ts_values[fidx]) if fidx < len(ts_values) else ts
                    gaze = pd_.get_gaze_at_timestamp(own_ts)
                    cfix = pd_.get_current_fixation(own_ts)
                    if gaze:
                        view.set_gaze(gaze[0], gaze[1])
                    else:
                        view.clear_gaze()
                    view.set_current_aoi(None)
                    view.set_fixations(pd_.get_fixations_at_timestamp(own_ts))
                    if self.aoi_manager:
                        aois_f = self.aoi_manager.get_all_aois_at_frame(pid, fidx)
                        colors = {n: self.aoi_manager.aois[n].color for n in aois_f}
                        is_kf = {n: self.aoi_manager.aois[n].has_keyframe_at(pid, fidx) for n in aois_f}
                        view.set_aois(aois_f, colors, is_kf)
                    # YOLO overlay for non-overlapping participant
                    if view.show_yolo and pd_.yolo_data:
                        view.set_yolo_boxes(get_detection_at_frame(pd_.yolo_data, fidx))
                    else:
                        view.set_yolo_boxes([])
                    view.update()
                    continue

            frame, fidx = pd_.get_frame_at_timestamp(ts)
            if frame is None and not hasattr(self, '_warned_no_frame'):
                print(f"  ⚠ {pid}: get_frame_at_timestamp returned None "
                      f"(ts={ts}, time_s={self.current_time_s:.3f}, fidx={fidx}, "
                      f"has_cap={pd_.video_cap is not None}, has_world={pd_.world is not None})")
                self._warned_no_frame = True
            view.set_frame(frame, fidx)
            gaze = pd_.get_gaze_at_timestamp(ts)
            cfix = pd_.get_current_fixation(ts)
            if gaze:
                view.set_gaze(gaze[0], gaze[1])
                if self.gaze_tracker:
                    is_f = cfix is not None; fd = cfix[2] if cfix else 0
                    aoi = self.gaze_tracker.analyze_gaze_in_aoi(
                        pid, ts, gaze[0], gaze[1], fidx, pd_.frame_width, pd_.frame_height, is_f, fd)
                    view.set_current_aoi(aoi)
                    view.set_scanpath(self.gaze_tracker.get_scanpath(pid))
            else:
                view.clear_gaze(); view.set_current_aoi(None)
            view.set_fixations(pd_.get_fixations_at_timestamp(ts))
            # AOIs
            if self.aoi_manager:
                aois_f = self.aoi_manager.get_all_aois_at_frame(pid, fidx)
                colors = {n: self.aoi_manager.aois[n].color for n in aois_f}
                is_kf = {n: self.aoi_manager.aois[n].has_keyframe_at(pid, fidx) for n in aois_f}
                view.set_aois(aois_f, colors, is_kf)
            # YOLO overlay
            if view.show_yolo and pd_.yolo_data:
                view.set_yolo_boxes(get_detection_at_frame(pd_.yolo_data, fidx))
            else:
                view.set_yolo_boxes([])
            view.update()
        dur = max(0, self.session.sync_info.overlap_duration_s)
        t = max(0, self.current_time_s)
        cm, cs = int(t//60), t%60
        dm, ds = int(dur//60), dur%60
        self.time_label.setText(f"{cm}:{cs:05.2f} / {dm}:{ds:05.2f}")
        self.timeline.set_current_time(self.current_time_s)

    def _update_opts(self):
        self._apply_view_opts()
        self._display_frame()

    def _apply_view_opts(self):
        sg = self.chk_gaze.isChecked(); sf = self.chk_fix.isChecked()
        st = self.chk_trail.isChecked(); ss = self.chk_scan.isChecked()
        sa = self.chk_aois.isChecked(); sy = self.chk_yolo.isChecked()
        for v in self.participant_views.values():
            v.show_gaze = sg; v.show_fixations = sf; v.show_trail = st
            v.show_scanpath = ss; v.show_aois = sa; v.show_yolo = sy
            if not st: v.gaze_trail.clear()

    def _update_aoi_panel(self):
        if self.aoi_manager:
            self.aoi_panel.set_aois(self.aoi_manager.aois)

    # ── Playback ─────────────────────────────────────────────────────────────

    def _toggle_play(self):
        if self.is_playing: self._pause()
        else: self._play()

    def _play(self):
        self.is_playing = True; self.btn_play.setText("⏸")
        fps = 30
        if self.session and self.session.participants:
            fps = np.mean([p.fps for p in self.session.participants.values()])
        self.play_timer.start(int(1000 / fps))

    def _pause(self):
        self.is_playing = False; self.btn_play.setText("▶"); self.play_timer.stop()

    def _advance_frame(self):
        if not self.session or not self.session.sync_info: 
            self._pause(); return
        dur = self.session.sync_info.overlap_duration_s
        if dur <= 0.01:
            self._pause()
            self.status_bar.showMessage("Cannot play: duration is 0. Check terminal for errors.")
            return
        fps = 30
        if self.session.participants:
            fps_values = [p.fps for p in self.session.participants.values() if p.fps > 0]
            fps = np.mean(fps_values) if fps_values else 30
        self.current_time_s += 1 / fps
        if self.current_time_s >= dur:
            self.current_time_s = dur; self._pause()
        self._display_frame()

    def _seek_time(self, t):
        if not self.session or not self.session.sync_info: return
        if self.gaze_tracker: self.gaze_tracker.clear_scanpaths()
        dur = max(0, self.session.sync_info.overlap_duration_s)
        self.current_time_s = max(0, min(t, dur))
        self._display_frame()

    def _seek_end(self):
        if self.session: self._seek_time(self.session.sync_info.overlap_duration_s)

    def _step(self, n):
        if not self.session: return
        fps = np.mean([p.fps for p in self.session.participants.values()]) if self.session.participants else 30
        self.current_time_s = max(0, min(self.current_time_s + n/fps,
                                         self.session.sync_info.overlap_duration_s))
        self._display_frame()

    # ── AOI Management ───────────────────────────────────────────────────────

    def _start_draw_first(self):
        if self.participant_views:
            self._start_draw(next(iter(self.participant_views)))

    def _start_draw(self, pid):
        for v in self.participant_views.values(): v.stop_drawing_aoi()
        self.drawing_participant = pid
        view = self.participant_views.get(pid)
        if view:
            view.start_drawing_aoi()
            self.status_bar.showMessage(f"Drawing AOI on {pid.upper()} — drag to create box")

    def _cancel_draw(self):
        for v in self.participant_views.values(): v.stop_drawing_aoi()
        self.drawing_participant = None; self.status_bar.showMessage("Drawing cancelled")

    def _delete_selected_keyframe(self):
        """Delete keyframe at current frame for any selected AOI across all views."""
        for v in self.participant_views.values():
            v.delete_selected_keyframe()

    def _on_aoi_created(self, name, bbox):
        if not self.aoi_manager or not self.session: return
        try:
            ts = self.session.get_current_timestamp(self.current_time_s)
            pfidx = {}
            for pid, pd_ in self.session.participants.items():
                if pd_.world is not None:
                    idx = int(np.searchsorted(pd_.world['timestamp [ns]'].values, ts))
                    pfidx[pid] = max(0, min(idx, pd_.total_frames - 1))
            self.aoi_manager.create_aoi(name, pfidx, bbox)
            self._update_aoi_panel(); self._display_frame()
            self.status_bar.showMessage(f"Created AOI '{name}' for all participants")
        except ValueError as e:
            QMessageBox.warning(self, "Error", str(e))
        self.drawing_participant = None

    def _on_aoi_updated(self, name, pid, fidx, bbox):
        if not self.aoi_manager: return
        try:
            self.aoi_manager.update_aoi_isolated(name, pid, fidx, bbox)
            self._update_aoi_panel(); self._display_frame()
            self.status_bar.showMessage(f"Updated '{name}' for {pid} at frame {fidx}")
        except ValueError as e:
            QMessageBox.warning(self, "Error", str(e))

    def _on_aoi_selected(self, name):
        for v in self.participant_views.values():
            v.selected_aoi = name; v.update()

    def _on_aoi_deleted(self, name):
        if self.aoi_manager:
            self.aoi_manager.delete_aoi(name)
            self._update_aoi_panel(); self._display_frame()

    def _on_delete_all_aois(self):
        if not self.aoi_manager or not self.aoi_manager.aois:
            return
        n = len(self.aoi_manager.aois)
        r = QMessageBox.question(
            self, "Delete All AOIs",
            f"Delete all {n} AOI(s)? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if r != QMessageBox.StandardButton.Yes:
            return
        for name in list(self.aoi_manager.aois.keys()):
            self.aoi_manager.delete_aoi(name)
        self._update_aoi_panel()
        self._display_frame()
        self.status_bar.showMessage(f"Deleted all {n} AOI(s)")

    def _on_aoi_keyframe_deleted(self, name, pid, fidx):
        if not self.aoi_manager: return
        try:
            self.aoi_manager.delete_keyframe(name, pid, fidx)
            self._update_aoi_panel(); self._display_frame()
            self.status_bar.showMessage(f"Deleted keyframe for '{name}' ({pid}) at frame {fidx}")
        except ValueError as e:
            QMessageBox.warning(self, "Error", str(e))

    def _on_aoi_renamed(self, old, new):
        if self.aoi_manager:
            try:
                self.aoi_manager.rename_aoi(old, new)
                self._update_aoi_panel(); self._display_frame()
            except ValueError as e:
                QMessageBox.warning(self, "Error", str(e))

    # ── YOLO Import ──────────────────────────────────────────────────────────

    def _import_yolo(self):
        if not self.session:
            QMessageBox.information(self, "No Session", "Load a session first."); return
        all_tracks = {}
        all_data = {}
        for pid, pd_ in self.session.participants.items():
            if pd_.yolo_data and pd_.yolo_data.get('tracks'):
                all_tracks[pid] = pd_.yolo_data['tracks']
                all_data[pid] = pd_.yolo_data
        if not all_tracks:
            QMessageBox.information(self, "No YOLO Data",
                "No YOLO detections found. Run:\n"
                "  gazeplayer yolo /path/to/session\n"
                "to generate detection data first.")
            return
        dialog = YOLOImportDialog(all_tracks, all_data,
                                  self.config.presets_for(list(self.session.participants)), self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            imports = dialog.get_imports()
            if not imports:
                return
            self.status_bar.showMessage(f"Importing {len(imports)} tracks...")
            QApplication.processEvents()
            count = 0
            for pid, track_id, aoi_name in imports:
                pd_ = self.session.participants.get(pid)
                if pd_ and pd_.yolo_data:
                    aoi = self.aoi_manager.create_from_yolo_track(
                        aoi_name, pd_.yolo_data, track_id, pid, auto_save=False)
                    if aoi: count += 1
            # Single save at the end
            self.aoi_manager.save()
            self._update_aoi_panel(); self._display_frame()
            self.status_bar.showMessage(f"Imported {count} YOLO tracks as AOIs")

    def _open_track_manager(self):
        """Switch to the Tracks tab in the sidebar and enable YOLO overlay."""
        if not self.session:
            QMessageBox.information(self, "No Session", "Load a session first."); return
        has_yolo = any(pd_.yolo_data and pd_.yolo_data.get('tracks')
                       for pd_ in self.session.participants.values())
        if not has_yolo:
            QMessageBox.information(self, "No YOLO Data",
                "No YOLO detections found. Run:\n"
                "  gazeplayer yolo /path/to/session\n"
                "to generate detection data first.")
            return

        # Enable YOLO overlay
        if not self.chk_yolo.isChecked():
            self.chk_yolo.setChecked(True)

        # Switch to Tracks tab (index 1)
        self.sidebar_tabs.setCurrentIndex(1)
        self.status_bar.showMessage(
            "Tracks tab — scrub video to inspect, select tracks to merge/delete")

    def _on_tracks_changed(self):
        """Refresh display after merge/delete in track manager."""
        self._display_frame()

    def _on_highlight_track(self, pid, track_id):
        """Highlight a specific track in the video views."""
        for p, view in self.participant_views.items():
            if p == pid and track_id is not None:
                view.highlighted_yolo_track = track_id
            else:
                view.highlighted_yolo_track = None
            view.update()

    def _on_seek_to_frame(self, pid, frame_idx):
        """Seek the player to a specific frame index for a participant."""
        pd_ = self.session.participants.get(pid)
        if not pd_ or pd_.world is None or not self.session.sync_info:
            return
        
        # Check if this participant is non-overlapping (timestamps outside sync window)
        ts_values = pd_.world['timestamp [ns]'].values
        is_non_overlapping = (ts_values[0] > self.session.sync_info.overlap_end_ns or
                              ts_values[-1] < self.session.sync_info.overlap_start_ns)
        
        if is_non_overlapping:
            # For non-overlapping participants, map frame directly to time using FPS
            fps = pd_.fps if pd_.fps > 0 else 30
            t_s = frame_idx / fps
        elif frame_idx < len(pd_.world):
            ts_ns = int(pd_.world['timestamp [ns]'].iloc[frame_idx])
            t_s = (ts_ns - self.session.sync_info.overlap_start_ns) / 1e9
            t_s = max(0, min(t_s, self.session.sync_info.overlap_duration_s))
        else:
            fps = pd_.fps if pd_.fps > 0 else 30
            t_s = frame_idx / fps
        self._seek_time(t_s)

    def _on_import_tracks(self):
        """Import all remaining YOLO tracks as AOIs."""
        if not self.track_manager or not self.aoi_manager:
            return
        imports = self.track_manager.get_all_remaining_tracks()
        if not imports:
            QMessageBox.information(self, "No Tracks", "No tracks to import.")
            return
        self.status_bar.showMessage(f"Importing {len(imports)} tracks...")
        QApplication.processEvents()
        count = 0
        for pid, track_id, aoi_name in imports:
            pd_ = self.session.participants.get(pid)
            if pd_ and pd_.yolo_data:
                aoi = self.aoi_manager.create_from_yolo_track(
                    aoi_name, pd_.yolo_data, track_id, pid, auto_save=False)
                if aoi: count += 1
        self.aoi_manager.save()
        self._update_aoi_panel(); self._display_frame()
        # Switch to AOIs tab to show results
        self.sidebar_tabs.setCurrentIndex(0)
        self.status_bar.showMessage(f"Imported {count} YOLO tracks as AOIs")

    # ── FastSAM Mode ─────────────────────────────────────────────────────────

    def _toggle_fastsam_mode(self):
        """Toggle click-to-segment mode."""
        any_active = any(v.fastsam_mode for v in self.participant_views.values())
        for v in self.participant_views.values():
            v.fastsam_mode = not any_active
            v.setCursor(Qt.CursorShape.CrossCursor if v.fastsam_mode else Qt.CursorShape.ArrowCursor)
        if not any_active:
            self.status_bar.showMessage("FastSAM mode ON — click on any video to segment+annotate")
        else:
            self.status_bar.showMessage("FastSAM mode OFF")

    def _on_fastsam_click(self, pid, frame_idx, fx, fy):
        """Handle a click in FastSAM mode — find mask at click point and create AOI."""
        pd_ = self.session.participants.get(pid)
        if not pd_ or not pd_.fastsam_data:
            QMessageBox.information(self, "No FastSAM Data",
                f"No FastSAM masks for {pid}. Run:\n"
                f"  gazeplayer fastsam /path/to/session\n"
                f"to generate segmentation data first.")
            return

        from ..detection.fastsam import get_mask_at_point, annotate_mask, save_fastsam_data

        mask = get_mask_at_point(pd_.fastsam_data, frame_idx, int(fx), int(fy))
        if not mask:
            self.status_bar.showMessage(f"No mask found at ({fx:.0f}, {fy:.0f}) for {pid}")
            return

        bbox_tuple = mask['bbox']
        # Ask for AOI name
        presets = self.config.presets_for(list(self.session.participants)) + ["custom..."]
        name, ok = QInputDialog.getItem(self, "Label Mask", "Assign AOI name:", presets, 0, True)
        if not ok or not name: return
        if name == "custom...":
            name, ok = QInputDialog.getText(self, "AOI Name", "Enter name:")
            if not ok or not name: return

        # Annotate in FastSAM data
        annotate_mask(pd_.fastsam_data, frame_idx, mask['mask_id'], name)
        fsp = pd_.data_dir / 'fastsam_masks.json'
        save_fastsam_data(pd_.fastsam_data, fsp)

        # Create/update AOI from mask bbox
        x, y, w, h = bbox_tuple
        bbox = BoundingBox(float(x), float(y), float(w), float(h))
        try:
            ts = self.session.get_current_timestamp(self.current_time_s)
            pfidx = {}
            for p, pd2 in self.session.participants.items():
                if pd2.world is not None:
                    idx = int(np.searchsorted(pd2.world['timestamp [ns]'].values, ts))
                    pfidx[p] = max(0, min(idx, pd2.total_frames - 1))

            if name in self.aoi_manager.aois:
                self.aoi_manager.update_aoi(name, pid, frame_idx, bbox)
            else:
                self.aoi_manager.create_aoi(name, pfidx, bbox)
        except ValueError:
            self.aoi_manager.update_aoi(name, pid, frame_idx, bbox)

        self._update_aoi_panel(); self._display_frame()
        self.status_bar.showMessage(f"FastSAM: created/updated AOI '{name}' for {pid} at frame {frame_idx}")

    # ── Analysis ─────────────────────────────────────────────────────────────

    def _analyze(self):
        if not self.session or not self.aoi_manager or not self.gaze_tracker: return
        if not self.aoi_manager.aois:
            QMessageBox.information(self, "No AOIs", "Create AOIs first."); return
        MetricsDialog(self.gaze_tracker.compute_all_metrics(), self).exec()

    def _export_metrics(self):
        """Export gaze-AOI metrics directly to CSV files."""
        if not self.session or not self.aoi_manager or not self.gaze_tracker:
            return
        if not self.aoi_manager.aois:
            QMessageBox.information(self, "No AOIs", "Create AOIs first.")
            return
        
        metrics = self.gaze_tracker.compute_all_metrics()
        if not metrics:
            QMessageBox.warning(self, "No Data", "No participant data available.")
            return

        # Ask for output directory
        out_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", str(self.session.session_dir))
        if not out_dir:
            return
        out_path = Path(out_dir)
        export_metrics_csv(metrics, out_path)

        self.status_bar.showMessage(
            f"Exported metrics to {out_path}: dwell_time, transitions, summary", 5000)
        QMessageBox.information(self, "Export Complete",
            "Metrics exported to:\n\n" +
            "\n".join(f"• {out_path / name}" for name in METRICS_FILES))

    def _gen_transitions(self):
        if not self.session or not self.aoi_manager or not self.gaze_tracker: return
        if not self.aoi_manager.aois:
            QMessageBox.information(self, "No AOIs", "Create AOIs first."); return
        r = QMessageBox.question(self, "Generate Events",
            "Scan all gaze data for AOI transitions?\nExisting transitions will be replaced.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if r != QMessageBox.StandardButton.Yes: return
        self.session.clear_transition_events()
        prog = QProgressDialog("Generating...", "Cancel", 0, 100, self)
        prog.setWindowModality(Qt.WindowModality.WindowModal); prog.show()
        def cb(msg, pct): prog.setLabelText(msg); prog.setValue(pct); QApplication.processEvents()
        events = self.gaze_tracker.generate_all_transition_events(cb)
        prog.close()
        if events:
            self.session.add_events_batch(events)
            self.timeline.set_events(self.session.get_all_events_for_timeline())
            QMessageBox.information(self, "Done", f"Generated {len(events)} transition events.")
        else:
            QMessageBox.information(self, "No Transitions", "No AOI transitions found.")

    def _clear_transitions(self):
        if not self.session: return
        r = QMessageBox.question(self, "Clear Transition Events",
            "Remove all generated gaze transition events?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if r != QMessageBox.StandardButton.Yes: return
        self.session.clear_transition_events()
        self.timeline.set_events(self.session.get_all_events_for_timeline())
        self.status_bar.showMessage("Cleared all transition events.")

    # ── Events ───────────────────────────────────────────────────────────────

    def _add_event(self):
        if not self.session: return
        ts = self.session.get_current_timestamp(self.current_time_s)
        d = AddEventDialog(ts, self.session.sync_info.overlap_start_ns,
                           participants=list(self.session.participants),
                           event_types=self.config.event_types, parent=self)
        if d.exec() == QDialog.DialogCode.Accepted:
            data = d.get_event_data()
            if data['name']:
                self.session.add_event(data)
                self.timeline.set_events(self.session.get_all_events_for_timeline())
                self.status_bar.showMessage(f"Added event: {data['name']}")

    def closeEvent(self, event):
        if self.session: self.session.close()
        super().closeEvent(event)
