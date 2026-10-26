"""Modal dialogs: metrics viewer, event editor, YOLO track import."""

from typing import Dict, List, Optional, Sequence, Tuple

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QDialog,
    QDialogButtonBox, QTextEdit, QComboBox, QCheckBox, QGroupBox, QTabWidget,
    QTableWidget, QTableWidgetItem,
)

from ..analysis import GazeAOIMetrics
from ..config import DEFAULT_EVENT_TYPES


class MetricsDialog(QDialog):
    def __init__(self, metrics: Dict[str, GazeAOIMetrics], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gaze-AOI Metrics"); self.setMinimumSize(700, 500)
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        for pid, m in metrics.items():
            tab = QWidget(); tl = QVBoxLayout(tab)
            dg = QGroupBox("Dwell Time"); dl = QVBoxLayout(dg)
            dt = QTableWidget(); dt.setColumnCount(3)
            dt.setHorizontalHeaderLabels(["AOI", "Time (ms)", "% Total"])
            total = sum(m.dwell_time_ms.values()) or 1
            data = sorted(m.dwell_time_ms.items(), key=lambda x: -x[1])
            dt.setRowCount(len(data))
            for i, (aoi, ms) in enumerate(data):
                dt.setItem(i, 0, QTableWidgetItem(aoi))
                dt.setItem(i, 1, QTableWidgetItem(f"{ms:.1f}"))
                dt.setItem(i, 2, QTableWidgetItem(f"{100*ms/total:.1f}%"))
            dt.horizontalHeader().setStretchLastSection(True); dl.addWidget(dt); tl.addWidget(dg)
            if m.transitions:
                tg = QGroupBox("Transitions"); tgl = QVBoxLayout(tg)
                tt = QTableWidget(); tt.setColumnCount(3)
                tt.setHorizontalHeaderLabels(["From", "To", "Count"])
                td = sorted(m.transitions.items(), key=lambda x: -x[1])
                tt.setRowCount(len(td))
                for i, ((f_, t_), c) in enumerate(td):
                    tt.setItem(i, 0, QTableWidgetItem(f_))
                    tt.setItem(i, 1, QTableWidgetItem(t_))
                    tt.setItem(i, 2, QTableWidgetItem(str(c)))
                tt.horizontalHeader().setStretchLastSection(True); tgl.addWidget(tt); tl.addWidget(tg)
            tabs.addTab(tab, pid.upper())
        layout.addWidget(tabs)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        bb.accepted.connect(self.accept); layout.addWidget(bb)


# ── Event Dialog ─────────────────────────────────────────────────────────────

class AddEventDialog(QDialog):
    def __init__(self, ts_ns, overlap_start_ns, name="", etype="annotation",
                 participant="all", notes="", participants: Sequence[str] = (),
                 event_types: Optional[Sequence[str]] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Event" if not name else "Edit Event")
        self.setModal(True); self.setMinimumWidth(400)
        layout = QVBoxLayout(self); layout.setSpacing(12)
        nl = QHBoxLayout(); nl.addWidget(QLabel("Name:"))
        self.name_edit = QLineEdit(); self.name_edit.setText(name); nl.addWidget(self.name_edit)
        layout.addLayout(nl)
        tl = QHBoxLayout(); tl.addWidget(QLabel("Type:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems(list(event_types or DEFAULT_EVENT_TYPES))
        self.type_combo.setCurrentText(etype); self.type_combo.setEditable(True)
        tl.addWidget(self.type_combo); layout.addLayout(tl)
        pl = QHBoxLayout(); pl.addWidget(QLabel("Participant:"))
        self.part_combo = QComboBox()
        self.part_combo.addItems(["all"] + list(participants))
        self.part_combo.setCurrentText(participant); pl.addWidget(self.part_combo)
        layout.addLayout(pl)
        st = (ts_ns - overlap_start_ns) / 1e9
        layout.addWidget(QLabel(f"Time: {int(st//60)}:{st%60:05.2f}"))
        layout.addWidget(QLabel("Notes:"))
        self.notes_edit = QTextEdit(); self.notes_edit.setText(notes)
        self.notes_edit.setMaximumHeight(80); layout.addWidget(self.notes_edit)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject); layout.addWidget(bb)
        self.timestamp_ns = ts_ns; self.name_edit.setFocus()

    def get_event_data(self):
        return {
            'timestamp [ns]': self.timestamp_ns,
            'name': self.name_edit.text().strip(),
            'type': self.type_combo.currentText().strip() or 'annotation',
            'participant': self.part_combo.currentText(),
            'notes': self.notes_edit.toPlainText().strip(),
        }


# ── YOLO Import Dialog ──────────────────────────────────────────────────────

class YOLOImportDialog(QDialog):
    """Dialog for selecting which YOLO tracks to import as AOIs."""
    def __init__(self, all_tracks: Dict[str, Dict[str, dict]],
                 all_data: Dict[str, dict] = None,
                 presets: Sequence[str] = (), parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import YOLO Tracks as AOIs")
        self.setMinimumSize(750, 500)
        self.selected_imports = []
        self.presets = list(presets)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select YOLO tracks to import as AOIs.\n"
                                "Uncheck short/noisy tracks. Rename to match preset AOI names."))

        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(["Import", "Participant", "Track",
                                              "Class", "Frames", "Duration", "AOI Name"])
        self.table.horizontalHeader().setStretchLastSection(True)

        rows = []
        for pid, tracks in all_tracks.items():
            fps = 30.0
            if all_data and pid in all_data:
                fps = all_data[pid].get('fps', 30.0)
            for tid, info in sorted(tracks.items(), key=lambda x: int(x[0])):
                first = info.get('first_frame', 0)
                last = info.get('last_frame', 0)
                span_frames = last - first
                dur_s = span_frames / fps if fps > 0 else 0

                # Count actual detection frames
                det_count = 0
                if all_data and pid in all_data:
                    for fidx_str, dets in all_data[pid].get('frames', {}).items():
                        for d in dets:
                            if d.get('track_id') == int(tid):
                                det_count += 1
                                break
                rows.append((pid, tid, info, det_count, dur_s, fps))

        self.table.setRowCount(len(rows))
        self.checkboxes = []
        self.name_edits = []

        for i, (pid, tid, info, det_count, dur_s, fps) in enumerate(rows):
            # Auto-uncheck short tracks (likely false positives)
            is_short = det_count < 20
            cb = QCheckBox()
            cb.setChecked(not is_short)
            self.checkboxes.append(cb)
            self.table.setCellWidget(i, 0, cb)

            self.table.setItem(i, 1, QTableWidgetItem(pid))
            self.table.setItem(i, 2, QTableWidgetItem(str(tid)))
            cls_item = QTableWidgetItem(info.get('class', '?'))
            self.table.setItem(i, 3, cls_item)
            self.table.setItem(i, 4, QTableWidgetItem(str(det_count)))
            self.table.setItem(i, 5, QTableWidgetItem(f"{dur_s:.1f}s"))

            suggested = info.get('suggested_aoi', info.get('class', f'track_{tid}'))
            name_edit = QLineEdit(suggested)
            if is_short:
                name_edit.setStyleSheet("color: #888;")
            self.name_edits.append(name_edit)
            self.table.setCellWidget(i, 6, name_edit)

        self.table.resizeColumnsToContents()
        layout.addWidget(self.table)

        # Info label
        n_checked = sum(1 for cb in self.checkboxes if cb.isChecked())
        n_total = len(rows)
        self.info_label = QLabel(f"  {n_checked}/{n_total} tracks selected "
                                 f"(short tracks auto-unchecked)")
        self.info_label.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(self.info_label)

        # Preset assignment
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Assign presets:"))
        for preset in self.presets[:7]:
            btn = QPushButton(preset)
            btn.setFixedHeight(24)
            btn.setStyleSheet("font-size:10px;padding:2px 4px;")
            btn.clicked.connect(lambda _, p=preset: self._assign_preset(p))
            preset_row.addWidget(btn)
        preset_row.addStretch()
        layout.addLayout(preset_row)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        layout.addWidget(bb)
        self._rows = [(pid, tid, info) for pid, tid, info, _, _, _ in rows]

    def _assign_preset(self, preset_name):
        """Assign preset name to first selected row without a preset."""
        for i, (cb, ne) in enumerate(zip(self.checkboxes, self.name_edits)):
            if cb.isChecked() and ne.text() == ne.text():
                # If name is still the default suggestion, replace with preset
                if not any(ne.text() == p for p in self.presets):
                    ne.setText(preset_name)
                    return

    def get_imports(self) -> List[Tuple[str, int, str]]:
        result = []
        for i, (pid, tid, info) in enumerate(self._rows):
            if self.checkboxes[i].isChecked():
                name = self.name_edits[i].text().strip()
                if name:
                    result.append((pid, int(tid), name))
        return result
