"""Sidebar listing AOIs plus the analysis/export buttons."""

from typing import Dict

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QMessageBox,
    QListWidget, QListWidgetItem, QInputDialog, QMenu,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor

from ..aoi import AOI
from .theme import Colors


class AOIPanel(QWidget):
    create_aoi_clicked = pyqtSignal()
    import_yolo_clicked = pyqtSignal()
    manage_tracks_clicked = pyqtSignal()
    fastsam_clicked = pyqtSignal()
    aoi_selected = pyqtSignal(str)
    aoi_deleted = pyqtSignal(str)
    aoi_renamed = pyqtSignal(str, str)
    delete_all_aois_clicked = pyqtSignal()
    analyze_clicked = pyqtSignal()
    export_metrics_clicked = pyqtSignal()
    gen_events_clicked = pyqtSignal()
    clear_events_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background-color: {Colors.BG_PANEL};")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        title = QLabel("📦 Areas of Interest")
        title.setStyleSheet("color: white; font-weight: bold; font-size: 12px;")
        layout.addWidget(title)

        # Buttons row 1
        btn_row = QHBoxLayout()
        self.add_btn = QPushButton("+ Draw")
        self.add_btn.setStyleSheet(f"background-color:{Colors.ACCENT};color:white;border:none;"
                                   f"font-size:11px;font-weight:bold;padding:4px 8px;border-radius:4px;")
        self.add_btn.clicked.connect(self.create_aoi_clicked.emit)
        btn_row.addWidget(self.add_btn)

        self.yolo_btn = QPushButton("🔍 YOLO")
        self.yolo_btn.setToolTip("Import AOIs from YOLO auto-detection")
        self.yolo_btn.setStyleSheet(f"background-color:#2d2d2d;color:#00ff88;border:1px solid {Colors.BORDER};"
                                    f"font-size:10px;padding:4px 6px;border-radius:4px;")
        self.yolo_btn.clicked.connect(self.import_yolo_clicked.emit)
        btn_row.addWidget(self.yolo_btn)

        self.sam_btn = QPushButton("✂ SAM")
        self.sam_btn.setToolTip("Click-to-segment with FastSAM (CPU)")
        self.sam_btn.setStyleSheet(f"background-color:#2d2d2d;color:#ff88ff;border:1px solid {Colors.BORDER};"
                                   f"font-size:10px;padding:4px 6px;border-radius:4px;")
        self.sam_btn.clicked.connect(self.fastsam_clicked.emit)
        btn_row.addWidget(self.sam_btn)
        layout.addLayout(btn_row)

        # Buttons row 2: Track manager
        btn_row2 = QHBoxLayout()
        self.manage_btn = QPushButton("🔧 Manage Tracks")
        self.manage_btn.setToolTip("Review, merge, and delete YOLO tracks while scrubbing video")
        self.manage_btn.setStyleSheet(f"background-color:#2d2d2d;color:#ffaa33;border:1px solid {Colors.BORDER};"
                                      f"font-size:10px;padding:4px 8px;border-radius:4px;")
        self.manage_btn.clicked.connect(self.manage_tracks_clicked.emit)
        btn_row2.addWidget(self.manage_btn)
        layout.addLayout(btn_row2)

        info = QLabel("Per-participant keyframes.\nHolds until next keyframe.")
        info.setStyleSheet(f"color:{Colors.TEXT_DIM};font-size:10px;"); info.setWordWrap(True)
        layout.addWidget(info)

        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet(f"""
            QListWidget {{background-color:{Colors.BG_ANNOTATION};border:1px solid {Colors.BORDER};border-radius:4px;}}
            QListWidget::item {{padding:6px;border-bottom:1px solid {Colors.BORDER};}}
            QListWidget::item:selected {{background-color:rgba(10,132,255,0.3);}}
        """)
        self.list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self._ctx_menu)
        self.list_widget.itemClicked.connect(lambda i: self.aoi_selected.emit(i.data(Qt.ItemDataRole.UserRole)))
        layout.addWidget(self.list_widget)

        bstyle = f"QPushButton{{background-color:#2d2d2d;color:white;border:1px solid {Colors.BORDER};" \
                 f"font-size:11px;padding:6px;border-radius:4px;}}QPushButton:hover{{background-color:#3d3d3d;}}"
        self.analyze_btn = QPushButton("📊 Compute Metrics")
        self.analyze_btn.setStyleSheet(bstyle)
        self.analyze_btn.clicked.connect(self.analyze_clicked.emit)
        layout.addWidget(self.analyze_btn)

        self.export_btn = QPushButton("💾 Export Metrics to CSV")
        self.export_btn.setStyleSheet(bstyle)
        self.export_btn.clicked.connect(self.export_metrics_clicked.emit)
        layout.addWidget(self.export_btn)

        self.gen_btn = QPushButton("🔄 Generate Transition Events")
        self.gen_btn.setStyleSheet(bstyle)
        self.gen_btn.clicked.connect(self.gen_events_clicked.emit)
        layout.addWidget(self.gen_btn)

        self.clear_events_btn = QPushButton("🗑 Clear Transition Events")
        self.clear_events_btn.setStyleSheet(bstyle)
        self.clear_events_btn.clicked.connect(self.clear_events_clicked.emit)
        layout.addWidget(self.clear_events_btn)

        self.delete_all_btn = QPushButton("🗑 Delete All AOIs")
        self.delete_all_btn.setStyleSheet(
            f"QPushButton{{background-color:#5a2d2d;color:white;border:1px solid #7a3d3d;"
            f"font-size:11px;padding:6px;border-radius:4px;}}"
            f"QPushButton:hover{{background-color:#7a3d3d;}}")
        self.delete_all_btn.clicked.connect(self.delete_all_aois_clicked.emit)
        layout.addWidget(self.delete_all_btn)

    def set_aois(self, aois: Dict[str, AOI]):
        self.list_widget.clear()
        for name, aoi in aois.items():
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, name)
            total_kf = sum(len(kfs) for kfs in aoi.participant_keyframes.values())
            item.setText(f"● {name} ({total_kf} kf)")
            item.setForeground(QColor(aoi.color))
            self.list_widget.addItem(item)

    def _ctx_menu(self, pos):
        item = self.list_widget.itemAt(pos)
        if not item: return
        name = item.data(Qt.ItemDataRole.UserRole)
        menu = QMenu(self)
        menu.setStyleSheet(f"QMenu{{background-color:{Colors.BG_PANEL};color:white;"
                           f"border:1px solid {Colors.BORDER};}}"
                           f"QMenu::item:selected{{background-color:{Colors.ACCENT};}}")
        ren = menu.addAction("Rename"); dl = menu.addAction("Delete")
        action = menu.exec(self.list_widget.mapToGlobal(pos))
        if action == ren:
            new, ok = QInputDialog.getText(self, "Rename", "New name:", text=name)
            if ok and new and new != name: self.aoi_renamed.emit(name, new)
        elif action == dl:
            r = QMessageBox.question(self, "Delete", f"Delete '{name}' from all participants?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if r == QMessageBox.StandardButton.Yes: self.aoi_deleted.emit(name)
