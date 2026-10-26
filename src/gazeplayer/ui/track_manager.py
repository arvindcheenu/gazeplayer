"""Tracks tab: review, rename, merge and delete YOLO tracks before importing
them as AOIs."""

import json
from typing import Dict, List, Optional, Sequence, Set, Tuple

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QMessageBox, QTabWidget, QTreeWidget, QTreeWidgetItem,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QShortcut, QKeySequence

from .participant_view import ParticipantView
from .theme import Colors


class YOLOTrackManager(QWidget):
    """Sidebar widget for reviewing, merging, renaming, and deleting YOLO tracks.
    Groups tracks by class. Lives in the right panel alongside AOI panel."""

    tracks_changed = pyqtSignal()
    highlight_track = pyqtSignal(str, object)  # pid, track_id (or None)
    seek_to_frame = pyqtSignal(str, int)  # pid, frame_idx
    import_requested = pyqtSignal()  # user wants to import remaining tracks

    TRACK_COLORS = ParticipantView.TRACK_COLORS

    def __init__(self, quick_labels: Sequence[str] = (), parent=None):
        super().__init__(parent)
        self.session = None
        self.quick_labels: List[str] = []
        self._label_shortcuts: List[QShortcut] = []
        self._pending_edits: Dict[str, Dict[int, str]] = {}  # pid -> {tid -> new_name}
        self._pending_deletions: Dict[str, Set[int]] = {}  # pid -> {tid, ...}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("color: #aaa; font-size: 10px;")
        layout.addWidget(self.info_label)

        # Keyboard shortcuts for quick class renaming
        self._setup_shortcuts()
        self.set_quick_labels(quick_labels)

        # Custom tree that prevents arrow navigation from jumping out of a class group
        # when user reaches the first/last child. An external callback can be set on
        # `end_of_group_cb(direction, class_name)` to notify the UI.
        # TrackTree is defined after __init__ so the __init__ body remains contiguous.

        # Quick rename bar
        rename_row = QHBoxLayout(); rename_row.setSpacing(4)
        rename_row.addWidget(QLabel("Class:"))
        self.rename_input = QLineEdit()
        self.rename_input.setPlaceholderText("Type new class name, press Enter")
        self.rename_input.setStyleSheet(
            f"QLineEdit{{background-color:{Colors.BG_ANNOTATION};color:white;"
            f"border:1px solid {Colors.BORDER};border-radius:3px;"
            f"padding:4px 6px;font-size:11px;}}"
            f"QLineEdit:focus{{border:1px solid {Colors.ACCENT};}}")
        self.rename_input.returnPressed.connect(self._quick_rename)
        rename_row.addWidget(self.rename_input, 1)
        self.rename_apply_btn = QPushButton("↩ Apply")
        self.rename_apply_btn.setStyleSheet(
            f"QPushButton{{background-color:#2d4a5a;color:white;border:none;"
            f"font-weight:bold;padding:4px 8px;border-radius:3px;font-size:10px;}}"
            f"QPushButton:hover{{background-color:#3d6a7a;}}")
        self.rename_apply_btn.clicked.connect(self._quick_rename)
        rename_row.addWidget(self.rename_apply_btn)
        layout.addLayout(rename_row)

        # Per-participant tabs, each with a tree grouped by class
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(
            f"QTabWidget::pane{{border:1px solid {Colors.BORDER};}}"
            f"QTabBar::tab{{background:{Colors.BG_ANNOTATION};color:#aaa;"
            f"padding:4px 8px;border:1px solid {Colors.BORDER};font-size:10px;}}"
            f"QTabBar::tab:selected{{background:{Colors.BG_PANEL};color:white;}}")
        self.track_trees: Dict[str, QTreeWidget] = {}
        layout.addWidget(self.tabs, 1)

        # Action buttons — compact
        btn_row = QHBoxLayout(); btn_row.setSpacing(3)
        bstyle = ("QPushButton{{background-color:{bg};color:white;border:none;"
                  "font-weight:bold;padding:4px 6px;border-radius:3px;font-size:10px;}}"
                  "QPushButton:hover{{background-color:{hover};}}")

        self.apply_btn = QPushButton("✓ Apply changes")
        self.apply_btn.setStyleSheet(bstyle.format(bg="#2d4a5a", hover="#3d6a7a"))
        self.apply_btn.clicked.connect(self._apply_pending_changes)
        self.apply_btn.setEnabled(False)
        btn_row.addWidget(self.apply_btn)

        self.merge_btn = QPushButton("🔗 Merge")
        self.merge_btn.setStyleSheet(bstyle.format(bg="#2d5a2d", hover="#3d7a3d"))
        self.merge_btn.clicked.connect(self._merge_selected)
        btn_row.addWidget(self.merge_btn)

        self.delete_btn = QPushButton("🗑 Delete")
        self.delete_btn.setStyleSheet(bstyle.format(bg="#5a2d2d", hover="#7a3d3d"))
        self.delete_btn.clicked.connect(self._delete_selected)
        btn_row.addWidget(self.delete_btn)
        layout.addLayout(btn_row)

        self.import_btn = QPushButton("📦 Import as AOIs")
        self.import_btn.setStyleSheet(
            f"QPushButton{{background-color:{Colors.ACCENT};color:white;border:none;"
            f"font-weight:bold;padding:6px;border-radius:4px;font-size:10px;}}"
            f"QPushButton:hover{{background-color:#0066cc;}}")
        self.import_btn.clicked.connect(self.import_requested.emit)
        layout.addWidget(self.import_btn)

        self.status = QLabel("")
        self.status.setStyleSheet("color: #aaa; font-size: 10px;")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

    def set_session(self, session):
        """Called when a session is loaded or reloaded."""
        self.session = session
        self._pending_edits.clear()
        self._update_apply_button()
        self._build_tabs()

    # ── TrackTree: prevents arrow navigation from jumping out of a class group ───
    class TrackTree(QTreeWidget):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.end_of_group_cb = None

        def keyPressEvent(self, event):
            key = event.key()
            if key in (Qt.Key.Key_Down, Qt.Key.Key_Up):
                curr = self.currentItem()
                if curr is not None:
                    parent = curr.parent()
                    # If current is a child inside a group, check bounds
                    if parent is not None:
                        idx = parent.indexOfChild(curr)
                        if key == Qt.Key.Key_Down and idx == parent.childCount() - 1:
                            # reached end of this group's list — notify and swallow
                            if callable(self.end_of_group_cb):
                                cls = parent.data(0, Qt.ItemDataRole.UserRole + 1) or parent.text(0)
                                try:
                                    self.end_of_group_cb('down', cls)
                                except Exception:
                                    pass
                            return
                        if key == Qt.Key.Key_Up and idx == 0:
                            if callable(self.end_of_group_cb):
                                cls = parent.data(0, Qt.ItemDataRole.UserRole + 1) or parent.text(0)
                                try:
                                    self.end_of_group_cb('up', cls)
                                except Exception:
                                    pass
                            return
            super().keyPressEvent(event)

    def _on_end_of_group(self, direction: str, cls: str):
        """Called when user presses arrow at the boundary of a class group."""
        if direction == 'down':
            self.status.setText(f"⏹ End of '{cls}' group — click another group to continue")
        else:
            self.status.setText(f"⏹ Start of '{cls}' group — click another group to continue")

    # ── Build tree ───────────────────────────────────────────────────────────

    def _build_tabs(self):
        current_tab = self.tabs.currentIndex()
        self._building = True  # Suppress itemChanged during rebuild
        self.tabs.clear()
        self.track_trees.clear()
        if not self.session:
            self._building = False
            return

        for pid, pd_ in self.session.participants.items():
            if not pd_.yolo_data or not pd_.yolo_data.get('tracks'):
                continue
            tracks = pd_.yolo_data['tracks']
            frames = pd_.yolo_data.get('frames', {})
            fps = pd_.yolo_data.get('fps', 30)

            tree = self.TrackTree()
            tree.end_of_group_cb = self._on_end_of_group
            tree.setHeaderLabels(["Track", "Class", "Frames", "Duration", "Range"])
            tree.setSelectionMode(QTreeWidget.SelectionMode.ExtendedSelection)
            tree.setRootIsDecorated(True)
            tree.setIndentation(20)
            # Disable automatic edit triggers - we manually trigger edit only for column 1
            tree.setEditTriggers(QTreeWidget.EditTrigger.NoEditTriggers)
            # Enable sorting by clicking column headers
            tree.setSortingEnabled(True)
            tree.header().setSectionsClickable(True)
            tree.header().sectionClicked.connect(
                lambda col, p=pid: self._on_header_clicked(p, col))
            tree.setStyleSheet(
                f"QTreeWidget{{background-color:{Colors.BG_ANNOTATION};"
                f"border:1px solid {Colors.BORDER};border-radius:4px;color:white;}}"
                f"QTreeWidget::item{{padding:4px 2px;"
                f"border-bottom:1px solid {Colors.BORDER};}}"
                f"QTreeWidget::item:selected{{background-color:rgba(10,132,255,0.3);}}"
                f"QTreeWidget::branch{{background-color:{Colors.BG_ANNOTATION};}}")

            # Group tracks by class
            by_class: Dict[str, List[Tuple[int, dict, int]]] = {}
            for tid_str, tinfo in tracks.items():
                tid = int(tid_str)
                cls = tinfo.get('class', '?')
                det_count = sum(1 for dets in frames.values()
                                if any(d.get('track_id') == tid for d in dets))
                by_class.setdefault(cls, []).append((tid, tinfo, det_count))

            # Sort classes by total detections descending
            for cls in sorted(by_class, key=lambda c: sum(x[2] for x in by_class[c]),
                              reverse=True):
                class_items = by_class[cls]
                total_dets = sum(x[2] for x in class_items)

                # Class group header (not editable, not selectable)
                group = QTreeWidgetItem(tree)
                group.setText(0, f"{cls}")
                group.setText(1, f"({len(class_items)} tracks)")
                group.setText(2, str(total_dets))
                group.setFlags(group.flags() & ~Qt.ItemFlag.ItemIsSelectable
                               & ~Qt.ItemFlag.ItemIsEditable)
                font = group.font(0); font.setBold(True); font.setPointSize(font.pointSize() + 1)
                group.setFont(0, font)
                group.setForeground(0, QColor("#ffffff"))
                group.setExpanded(True)
                # Store class name for group operations
                group.setData(0, Qt.ItemDataRole.UserRole + 1, cls)

                # Individual tracks, sorted by det count desc
                for tid, tinfo, det_count in sorted(class_items, key=lambda x: -x[2]):
                    first = tinfo.get('first_frame', 0)
                    last = tinfo.get('last_frame', 0)
                    dur_s = (last - first) / fps if fps > 0 else 0

                    tc = self.TRACK_COLORS[tid % len(self.TRACK_COLORS)]
                    child = QTreeWidgetItem(group)
                    child.setText(0, f"T{tid}")
                    child.setText(1, cls)  # Editable class name
                    child.setText(2, str(det_count))
                    child.setText(3, f"{dur_s:.1f}s")
                    child.setText(4, f"f{first}–f{last}")
                    child.setData(0, Qt.ItemDataRole.UserRole, tid)
                    # Make only column 1 (Class) editable
                    child.setFlags(child.flags() | Qt.ItemFlag.ItemIsEditable)
                    child.setForeground(0, QColor(tc))
                    font = child.font(0); font.setBold(True); child.setFont(0, font)

                    if det_count < 20:
                        for col in range(5):
                            child.setForeground(col, QColor("#666666"))

            tree.setColumnWidth(0, 60)
            tree.setColumnWidth(1, 120)
            tree.setColumnWidth(2, 55)
            tree.setColumnWidth(3, 55)

            # Highlight on selection change (keyboard arrows or click) and auto-seek
            tree.currentItemChanged.connect(
                lambda curr, prev, p=pid: self._on_item_changed(p, curr, auto_seek=True))
            # Double-click: col 1 = edit in place (handled by tree); other cols = navigate
            tree.itemDoubleClicked.connect(
                lambda item, col, p=pid: self._on_item_double_clicked(p, item, col))
            # Inline edit finished
            tree.itemChanged.connect(
                lambda item, col, p=pid: self._on_item_edited(p, item, col))

            self.track_trees[pid] = tree
            n = len(tracks)
            self.tabs.addTab(tree, f"{pid} ({n})")

        # Restore tab
        if 0 <= current_tab < self.tabs.count():
            self.tabs.setCurrentIndex(current_tab)
        self._building = False

    def _on_item_changed(self, pid, item, auto_seek=False):
        if item and item.data(0, Qt.ItemDataRole.UserRole) is not None:
            tid = item.data(0, Qt.ItemDataRole.UserRole)
            self.highlight_track.emit(pid, tid)
            # Auto-seek to first frame when navigating with arrow keys or clicking
            if auto_seek:
                pd_ = self.session.participants.get(pid) if self.session else None
                if pd_ and pd_.yolo_data:
                    info = pd_.yolo_data.get('tracks', {}).get(str(tid), {})
                    first_frame = info.get('first_frame', 0)
                    self.seek_to_frame.emit(pid, first_frame)
                    self.status.setText(f"→ T{tid} at frame {first_frame}")
        else:
            self.highlight_track.emit(pid, None)

    def _on_header_clicked(self, pid, col):
        """Sort tracks within groups by the clicked column."""
        tree = self.track_trees.get(pid)
        if not tree:
            return
        pd_ = self.session.participants.get(pid) if self.session else None
        if not pd_ or not pd_.yolo_data:
            return

        # Determine sort order (toggle between ascending/descending)
        header = tree.header()
        current_order = header.sortIndicatorOrder()
        new_order = Qt.SortOrder.DescendingOrder if current_order == Qt.SortOrder.AscendingOrder else Qt.SortOrder.AscendingOrder
        
        # Sort children within each group
        for i in range(tree.topLevelItemCount()):
            group = tree.topLevelItem(i)
            if group.childCount() == 0:
                continue
            
            # Collect children with their sort keys
            children = []
            for j in range(group.childCount()):
                child = group.child(j)
                # Get sort key based on column
                if col == 0:  # Track ID
                    tid = child.data(0, Qt.ItemDataRole.UserRole)
                    key = tid if tid is not None else 0
                elif col == 2:  # Frames (numeric)
                    try:
                        key = int(child.text(2))
                    except ValueError:
                        key = 0
                elif col == 3:  # Duration (numeric, strip 's')
                    try:
                        key = float(child.text(3).rstrip('s'))
                    except ValueError:
                        key = 0.0
                elif col == 4:  # Range (sort by first frame)
                    range_text = child.text(4)
                    try:
                        first = int(range_text.split('–')[0].lstrip('f'))
                        key = first
                    except (ValueError, IndexError):
                        key = 0
                else:  # Text columns (Track, Class)
                    key = child.text(col).lower()
                children.append((key, child.clone()))
            
            # Sort and rebuild
            reverse = (new_order == Qt.SortOrder.DescendingOrder)
            children.sort(key=lambda x: x[0], reverse=reverse)
            
            # Remove existing children and re-add sorted
            while group.childCount() > 0:
                group.takeChild(0)
            for _, child_clone in children:
                group.addChild(child_clone)
        
        # Update sort indicator
        header.setSortIndicator(col, new_order)
        self.status.setText(f"Sorted by {tree.headerItem().text(col)} ({'desc' if reverse else 'asc'})")

    def _on_item_double_clicked(self, pid, item, col=None):
        tid = item.data(0, Qt.ItemDataRole.UserRole)
        if tid is None:
            return  # Clicked a group header
        # Column 1 = class name → manually start editing
        if col == 1:
            tree = self.track_trees.get(pid)
            if tree:
                tree.editItem(item, 1)
            return
        # Any other column → navigate to first frame
        pd_ = self.session.participants.get(pid)
        if not pd_ or not pd_.yolo_data:
            return
        info = pd_.yolo_data.get('tracks', {}).get(str(tid), {})
        first_frame = info.get('first_frame', 0)
        self.highlight_track.emit(pid, tid)
        self.seek_to_frame.emit(pid, first_frame)
        self.status.setText(f"→ T{tid} first detection at frame {first_frame}")

    def _on_item_edited(self, pid, item, col):
        """Handle inline class name edit (column 1) - stores as pending edit."""
        if self._building or col != 1:
            return
        tid = item.data(0, Qt.ItemDataRole.UserRole)
        if tid is None:
            return  # Group header, ignore

        new_name = item.text(1).strip()
        if not new_name:
            return

        pd_ = self.session.participants.get(pid)
        if not pd_ or not pd_.yolo_data:
            return

        data = pd_.yolo_data
        tracks = data.get('tracks', {})
        old_name = tracks.get(str(tid), {}).get('class', '?')

        if new_name == old_name:
            # Remove from pending if it was there
            if pid in self._pending_edits and tid in self._pending_edits[pid]:
                del self._pending_edits[pid][tid]
                if not self._pending_edits[pid]:
                    del self._pending_edits[pid]
            self._update_apply_button()
            return

        # Store as pending edit
        if pid not in self._pending_edits:
            self._pending_edits[pid] = {}
        self._pending_edits[pid][tid] = new_name
        self._update_apply_button()
        self.status.setText(f"Pending: T{tid} → '{new_name}' (click Apply changes to save)")

    def _update_apply_button(self):
        """Enable/disable Apply button based on pending edits and deletions."""
        has_edits = any(self._pending_edits.values())
        has_deletions = any(self._pending_deletions.values())
        has_pending = has_edits or has_deletions
        self.apply_btn.setEnabled(has_pending)
        if has_pending:
            edit_count = sum(len(edits) for edits in self._pending_edits.values())
            del_count = sum(len(dels) for dels in self._pending_deletions.values())
            parts = []
            if edit_count:
                parts.append(f"{edit_count} rename")
            if del_count:
                parts.append(f"{del_count} delete")
            self.apply_btn.setText(f"✓ Apply ({', '.join(parts)})")
        else:
            self.apply_btn.setText("✓ Apply changes")

    def set_quick_labels(self, labels: Sequence[str]):
        """Bind keys 1-9 to *labels* (stage a rename of the selected tracks)."""
        for sc in self._label_shortcuts:
            sc.setEnabled(False)
            sc.deleteLater()
        self.quick_labels = list(labels)[:9]
        self._label_shortcuts = []
        for i, name in enumerate(self.quick_labels, 1):
            shortcut = QShortcut(QKeySequence(str(i)), self)
            shortcut.activated.connect(lambda n=name: self._stage_rename_to(n))
            self._label_shortcuts.append(shortcut)
        keys = " ".join(f"{i}={n}" for i, n in enumerate(self.quick_labels, 1))
        self.info_label.setText("Select tracks → type new class → Enter. "
                                "Arrow keys to navigate & auto-jump.\n"
                                f"Keys: {keys + ', ' if keys else ''}Backspace=delete")

    def _setup_shortcuts(self):
        """Backspace marks the selected tracks for deletion."""
        del_shortcut = QShortcut(QKeySequence("Backspace"), self)
        del_shortcut.activated.connect(self._stage_deletion)

    def _stage_rename_to(self, new_name: str):
        """Stage a rename for all selected tracks to new_name (pending, not applied)."""
        pid = self._get_current_pid()
        if not pid:
            return
        pd_ = self.session.participants.get(pid) if self.session else None
        if not pd_ or not pd_.yolo_data:
            return

        tids = self._get_selected_tids()
        group_class = self._get_selected_class_group()

        # If a class group header is selected, rename all tracks in that class
        if group_class and not tids:
            tracks = pd_.yolo_data.get('tracks', {})
            tids = [int(tid_str) for tid_str, info in tracks.items()
                    if info.get('class', '?') == group_class]

        if not tids:
            self.status.setText(f"Select tracks first, then press 1-{len(self.quick_labels)}.")
            return

        tracks = pd_.yolo_data.get('tracks', {})

        # Stage each track as pending edit
        if pid not in self._pending_edits:
            self._pending_edits[pid] = {}

        staged_count = 0
        for tid in tids:
            old_name = tracks.get(str(tid), {}).get('class', '?')
            if new_name != old_name:
                self._pending_edits[pid][tid] = new_name
                staged_count += 1
                # Update the tree item visually
                tree = self.track_trees.get(pid)
                if tree:
                    self._update_item_text(tree, tid, new_name)

        self._update_apply_button()
        if staged_count:
            self.status.setText(f"Staged {staged_count} track(s) → '{new_name}' (click Apply to save)")
        else:
            self.status.setText(f"Selected tracks already named '{new_name}'")

    def _update_item_text(self, tree: QTreeWidget, tid: int, new_name: str):
        """Update the Class column text for a track item without triggering itemChanged."""
        self._building = True  # Suppress itemChanged
        for i in range(tree.topLevelItemCount()):
            group = tree.topLevelItem(i)
            for j in range(group.childCount()):
                child = group.child(j)
                if child.data(0, Qt.ItemDataRole.UserRole) == tid:
                    child.setText(1, new_name)
                    # Highlight pending items
                    child.setForeground(1, QColor("#ffcc00"))
                    self._building = False
                    return
        self._building = False

    def _stage_deletion(self):
        """Toggle deletion mark for selected tracks (applied on 'Apply changes')."""
        pid = self._get_current_pid()
        if not pid:
            return
        pd_ = self.session.participants.get(pid) if self.session else None
        if not pd_ or not pd_.yolo_data:
            return

        tids = self._get_selected_tids()
        group_class = self._get_selected_class_group()

        # If a class group header is selected, mark all tracks in that class
        if group_class and not tids:
            tracks = pd_.yolo_data.get('tracks', {})
            tids = [int(tid_str) for tid_str, info in tracks.items()
                    if info.get('class', '?') == group_class]

        if not tids:
            self.status.setText("Select tracks first, then press Backspace.")
            return

        if pid not in self._pending_deletions:
            self._pending_deletions[pid] = set()

        tree = self.track_trees.get(pid)
        marked_count = 0
        unmarked_count = 0

        for tid in tids:
            if tid in self._pending_deletions[pid]:
                # Already marked → unmark (toggle off)
                self._pending_deletions[pid].remove(tid)
                unmarked_count += 1
                if tree:
                    self._unmark_item_for_deletion(tree, tid)
            else:
                # Not marked → mark for deletion (toggle on)
                self._pending_deletions[pid].add(tid)
                marked_count += 1
                # Also remove from pending edits if exists (no point renaming deleted track)
                if pid in self._pending_edits and tid in self._pending_edits[pid]:
                    del self._pending_edits[pid][tid]
                if tree:
                    self._mark_item_for_deletion(tree, tid)

        # Clean up empty sets
        if pid in self._pending_deletions and not self._pending_deletions[pid]:
            del self._pending_deletions[pid]

        self._update_apply_button()
        if marked_count and unmarked_count:
            self.status.setText(f"Marked {marked_count}, unmarked {unmarked_count} track(s)")
        elif marked_count:
            self.status.setText(f"Marked {marked_count} track(s) for deletion (click Apply to confirm)")
        elif unmarked_count:
            self.status.setText(f"Unmarked {unmarked_count} track(s) from deletion")

    def _mark_item_for_deletion(self, tree: QTreeWidget, tid: int):
        """Visually mark a track item as pending deletion (red strikethrough style)."""
        self._building = True
        for i in range(tree.topLevelItemCount()):
            group = tree.topLevelItem(i)
            for j in range(group.childCount()):
                child = group.child(j)
                if child.data(0, Qt.ItemDataRole.UserRole) == tid:
                    # Red color and strikethrough font for all columns
                    font = child.font(0)
                    font.setStrikeOut(True)
                    for col in range(5):
                        child.setForeground(col, QColor("#ff5555"))
                        child.setFont(col, font)
                    self._building = False
                    return
        self._building = False

    def _unmark_item_for_deletion(self, tree: QTreeWidget, tid: int):
        """Remove deletion mark from a track item (restore normal style)."""
        self._building = True
        for i in range(tree.topLevelItemCount()):
            group = tree.topLevelItem(i)
            for j in range(group.childCount()):
                child = group.child(j)
                if child.data(0, Qt.ItemDataRole.UserRole) == tid:
                    # Remove strikethrough and restore normal color
                    font = child.font(0)
                    font.setStrikeOut(False)
                    for col in range(5):
                        child.setFont(col, font)
                        child.setForeground(col, QColor("white"))
                    # Restore track color for column 0
                    tc = self.TRACK_COLORS[tid % len(self.TRACK_COLORS)]
                    child.setForeground(0, QColor(tc))
                    self._building = False
                    return
        self._building = False

    def _apply_pending_changes(self):
        """Apply all pending class name edits and deletions."""
        if not self._pending_edits and not self._pending_deletions:
            return

        total_renamed = 0
        total_deleted_tracks = 0
        total_deleted_dets = 0

        # First apply renames
        for pid, edits in list(self._pending_edits.items()):
            pd_ = self.session.participants.get(pid)
            if not pd_ or not pd_.yolo_data:
                continue

            data = pd_.yolo_data
            tracks = data.get('tracks', {})

            for tid, new_name in edits.items():
                # Skip if also marked for deletion
                if pid in self._pending_deletions and tid in self._pending_deletions[pid]:
                    continue

                # Update track metadata
                if str(tid) in tracks:
                    tracks[str(tid)]['class'] = new_name
                    tracks[str(tid)]['suggested_aoi'] = new_name

                # Update frame-level detections
                for dets in data.get('frames', {}).values():
                    for det in dets:
                        if det.get('track_id') == tid:
                            det['class'] = new_name
                            if 'label_suggestion' in det:
                                det['label_suggestion'] = new_name
                            total_renamed += 1

            self._save_detections(pid, data)

        # Then apply deletions
        for pid, tids_to_delete in list(self._pending_deletions.items()):
            pd_ = self.session.participants.get(pid)
            if not pd_ or not pd_.yolo_data:
                continue

            data = pd_.yolo_data
            frames = data.get('frames', {})
            tracks = data.get('tracks', {})
            tids_set = set(tids_to_delete)

            # Remove from frames
            for fidx_str in list(frames.keys()):
                orig_len = len(frames[fidx_str])
                frames[fidx_str] = [d for d in frames[fidx_str]
                                    if d.get('track_id') not in tids_set]
                total_deleted_dets += orig_len - len(frames[fidx_str])
                if not frames[fidx_str]:
                    del frames[fidx_str]

            # Remove from tracks
            for tid in tids_to_delete:
                if tracks.pop(str(tid), None) is not None:
                    total_deleted_tracks += 1

            self._save_detections(pid, data)

        edit_count = sum(len(edits) for edits in self._pending_edits.values())
        del_count = sum(len(dels) for dels in self._pending_deletions.values())
        self._pending_edits.clear()
        self._pending_deletions.clear()
        self._update_apply_button()

        # Build status message
        parts = []
        if edit_count:
            parts.append(f"{edit_count} rename(s)")
        if total_deleted_tracks:
            parts.append(f"{total_deleted_tracks} track(s) deleted")
        self.status.setText(f"Applied: {', '.join(parts)}")
        self.tracks_changed.emit()
        # Rebuild to regroup
        self._build_tabs()

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _get_current_pid(self) -> Optional[str]:
        idx = self.tabs.currentIndex()
        if idx < 0:
            return None
        pids = list(self.track_trees.keys())
        return pids[idx] if idx < len(pids) else None

    def _get_selected_tids(self) -> List[int]:
        pid = self._get_current_pid()
        if not pid or pid not in self.track_trees:
            return []
        tree = self.track_trees[pid]
        tids = []
        for item in tree.selectedItems():
            tid = item.data(0, Qt.ItemDataRole.UserRole)
            if tid is not None:
                tids.append(tid)
        return tids

    def _get_selected_class_group(self) -> Optional[str]:
        """If a class group header is the current item, return its class name."""
        pid = self._get_current_pid()
        if not pid or pid not in self.track_trees:
            return None
        tree = self.track_trees[pid]
        item = tree.currentItem()
        if item and item.data(0, Qt.ItemDataRole.UserRole) is None and item.childCount() > 0:
            # This is a group header — class stored in UserRole+1
            cls = item.data(0, Qt.ItemDataRole.UserRole + 1)
            return cls if cls else item.text(0).strip()
        return None

    # ── Rename ───────────────────────────────────────────────────────────────

    def _quick_rename(self):
        """Rename all selected tracks (or all tracks in selected class group)
        to whatever is typed in the rename_input field."""
        new_name = self.rename_input.text().strip()
        if not new_name:
            self.status.setText("Type a class name first.")
            return

        pid = self._get_current_pid()
        if not pid:
            return
        pd_ = self.session.participants.get(pid)
        if not pd_ or not pd_.yolo_data:
            return

        tids = self._get_selected_tids()
        group_class = self._get_selected_class_group()

        # If a class group header is selected, rename all tracks in that class
        if group_class and not tids:
            tracks = pd_.yolo_data.get('tracks', {})
            tids = [int(tid_str) for tid_str, info in tracks.items()
                    if info.get('class', '?') == group_class]

        if not tids:
            self.status.setText("Select tracks or a class group first.")
            return

        data = pd_.yolo_data
        tracks = data.get('tracks', {})
        frames = data.get('frames', {})

        # Update track metadata
        for tid in tids:
            if str(tid) in tracks:
                tracks[str(tid)]['class'] = new_name
                tracks[str(tid)]['suggested_aoi'] = new_name

        # Update frame-level detections
        tids_set = set(tids)
        rename_count = 0
        for dets in frames.values():
            for det in dets:
                if det.get('track_id') in tids_set:
                    det['class'] = new_name
                    if 'label_suggestion' in det:
                        det['label_suggestion'] = new_name
                    rename_count += 1

        self._save_detections(pid, data)
        self._build_tabs()
        self.tracks_changed.emit()
        self.rename_input.clear()
        self.status.setText(
            f"Renamed {len(tids)} track(s) → '{new_name}' "
            f"({rename_count} detections updated)")

    # ── Merge ────────────────────────────────────────────────────────────────

    def _merge_selected(self):
        pid = self._get_current_pid()
        tids = self._get_selected_tids()
        if not pid or len(tids) < 2:
            QMessageBox.information(self, "Merge",
                                    "Select 2 or more tracks to merge.")
            return
        pd_ = self.session.participants.get(pid)
        if not pd_ or not pd_.yolo_data:
            return

        target_tid = min(tids)
        source_tids = [t for t in tids if t != target_tid]

        data = pd_.yolo_data
        frames = data.get('frames', {})
        tracks = data.get('tracks', {})

        # Reassign detections
        merged_count = 0
        for dets in frames.values():
            for det in dets:
                if det.get('track_id') in source_tids:
                    det['track_id'] = target_tid
                    merged_count += 1

        # Update track metadata
        target_info = tracks.get(str(target_tid), {})
        for src_tid in source_tids:
            src_info = tracks.get(str(src_tid), {})
            target_info['first_frame'] = min(
                target_info.get('first_frame', 999999),
                src_info.get('first_frame', 999999))
            target_info['last_frame'] = max(
                target_info.get('last_frame', 0),
                src_info.get('last_frame', 0))
            tracks.pop(str(src_tid), None)
        tracks[str(target_tid)] = target_info

        self._save_detections(pid, data)
        self._build_tabs()
        self.tracks_changed.emit()
        self.status.setText(
            f"Merged T{',T'.join(str(t) for t in source_tids)} → T{target_tid} "
            f"({merged_count} detections reassigned)")

    # ── Delete ───────────────────────────────────────────────────────────────

    def _delete_selected(self):
        pid = self._get_current_pid()
        tids = self._get_selected_tids()

        # Also support deleting an entire class group
        if not tids:
            group_class = self._get_selected_class_group()
            if group_class and pid:
                tracks = self.session.participants[pid].yolo_data.get('tracks', {})
                tids = [int(tid_str) for tid_str, info in tracks.items()
                        if info.get('class', '?') == group_class]

        if not pid or not tids:
            QMessageBox.information(self, "Delete",
                                    "Select tracks or a class group to delete.")
            return
        pd_ = self.session.participants.get(pid)
        if not pd_ or not pd_.yolo_data:
            return

        r = QMessageBox.question(self, "Delete Tracks",
            f"Delete {len(tids)} track(s) from {pid}?\n"
            f"Track IDs: {', '.join(f'T{t}' for t in sorted(tids))}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if r != QMessageBox.StandardButton.Yes:
            return

        data = pd_.yolo_data
        frames = data.get('frames', {})
        tracks = data.get('tracks', {})
        tids_set = set(tids)

        del_count = 0
        for fidx_str in list(frames.keys()):
            orig_len = len(frames[fidx_str])
            frames[fidx_str] = [d for d in frames[fidx_str]
                                if d.get('track_id') not in tids_set]
            del_count += orig_len - len(frames[fidx_str])
            if not frames[fidx_str]:
                del frames[fidx_str]

        for tid in tids:
            tracks.pop(str(tid), None)

        self._save_detections(pid, data)
        self._build_tabs()
        self.tracks_changed.emit()
        self.status.setText(
            f"Deleted {len(tids)} tracks ({del_count} detections removed)")

    # ── Save / Import ────────────────────────────────────────────────────────

    def _save_detections(self, pid, data):
        pd_ = self.session.participants.get(pid)
        if pd_:
            path = pd_.data_dir / 'yolo_detections.json'
            with open(path, 'w') as f:
                json.dump(data, f)
            pd_.yolo_data = data

    def _import_all(self):
        """Signal parent to import all remaining tracks as AOIs."""
        self.import_requested.emit()

    def get_all_remaining_tracks(self) -> List[Tuple[str, int, str]]:
        """Return all tracks still in the data for import.
        Uses class name as the AOI name (so merged/renamed classes group together)."""
        result = []
        for pid, pd_ in self.session.participants.items():
            if not pd_.yolo_data or not pd_.yolo_data.get('tracks'):
                continue
            for tid_str, info in pd_.yolo_data['tracks'].items():
                cls = info.get('class', '?')
                suggested = info.get('suggested_aoi', cls)
                result.append((pid, int(tid_str), suggested))
        return result
