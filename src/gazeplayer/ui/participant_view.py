"""One participant's video tile: frame, gaze overlays, AOI drawing/editing."""

from collections import deque
from enum import Enum, auto
from typing import Dict, List, Optional, Set

import numpy as np

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QInputDialog, QMenu
from PyQt6.QtCore import Qt, pyqtSignal, QPoint
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QPen

from ..aoi import BoundingBox
from ..analysis import ScanpathPoint
from .theme import Colors


class HandleType(Enum):
    NONE = auto(); MOVE = auto()
    TOP_LEFT = auto(); TOP_RIGHT = auto(); BOTTOM_LEFT = auto(); BOTTOM_RIGHT = auto()
    TOP = auto(); BOTTOM = auto(); LEFT = auto(); RIGHT = auto()


class ParticipantView(QWidget):
    """Single participant video view with gaze, AOI overlay, and drawing."""
    aoi_created = pyqtSignal(str, object)
    aoi_updated = pyqtSignal(str, str, int, object)
    aoi_selected = pyqtSignal(str)
    aoi_keyframe_deleted = pyqtSignal(str, str, int)  # aoi_name, pid, frame_idx
    fastsam_click = pyqtSignal(str, int, float, float)  # pid, frame_idx, x, y

    # Distinct colors per track ID
    TRACK_COLORS = [
        "#ff5733", "#33ff57", "#3357ff", "#ff33f5",
        "#f5ff33", "#33fff5", "#ff8c33", "#8c33ff",
        "#33ff8c", "#ff3333", "#ff6600", "#00ccff",
        "#cc00ff", "#00ff66", "#ffcc00", "#66ff00",
    ]

    def __init__(self, participant_id: str, color: str = Colors.GAZE,
                 aoi_presets: Optional[List[str]] = None, parent=None):
        super().__init__(parent)
        self.participant_id = participant_id
        self.color = color
        self.aoi_presets = list(aoi_presets or [])
        self.image: Optional[QPixmap] = None
        self.gaze_point = None
        self.fixations = []
        self.gaze_trail = deque(maxlen=30)
        self.scanpath: List[ScanpathPoint] = []
        self.current_aoi: Optional[str] = None
        self.show_gaze = True
        self.show_fixations = True
        self.show_trail = True
        self.show_scanpath = True
        self.show_aois = True
        self.show_yolo = False
        self.show_fastsam = False
        self.has_video = False
        self.current_frame_idx = 0
        self.aois: Dict[str, BoundingBox] = {}
        self.aoi_colors: Dict[str, str] = {}
        self.aoi_is_keyframe: Dict[str, bool] = {}  # True if current frame has actual keyframe
        self.selected_aoi: Optional[str] = None
        self.hovered_aoi: Optional[str] = None
        self.yolo_boxes: List[dict] = []  # Current frame YOLO detections
        self.hidden_yolo_tracks: Set[int] = set()  # Track IDs to hide
        self.highlighted_yolo_track: Optional[int] = None  # Track ID to highlight
        self.fastsam_mask_overlay: Optional[np.ndarray] = None
        # Drawing / manipulation state
        self.drawing_new_aoi = False
        self.draw_start: Optional[QPoint] = None
        self.draw_current: Optional[QPoint] = None
        self.manipulating = False
        self.handle_type = HandleType.NONE
        self.manip_start: Optional[QPoint] = None
        self.manip_original: Optional[BoundingBox] = None
        self.fastsam_mode = False
        self.frame_width = 1088
        self.frame_height = 1080
        self.setMinimumSize(320, 240)
        self.setMouseTracking(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(0)
        self.header = QLabel(participant_id.upper())
        self.header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.header.setStyleSheet(f"""
            background-color: {self.color}; color: black;
            font-weight: bold; font-size: 12px; padding: 4px;
            border-radius: 4px 4px 0 0;
        """)
        self.header.setFixedHeight(24)
        layout.addWidget(self.header)
        self.canvas = QWidget()
        self.canvas.setStyleSheet(f"background-color: transparent; border: 2px solid {self.color};")
        layout.addWidget(self.canvas, 1)

    def set_frame(self, frame, frame_idx=0):
        if frame is None:
            self.image = None
            self.has_video = False
            self.update()
            return
        self.has_video = True
        self.current_frame_idx = frame_idx
        self.frame_height, self.frame_width = frame.shape[:2]
        h, w, ch = frame.shape
        q_img = QImage(frame.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        self.image = QPixmap.fromImage(q_img)
        self.update()

    def set_gaze(self, x, y):
        self.gaze_point = (x, y)
        if self.show_trail:
            self.gaze_trail.append((x, y))

    def clear_gaze(self):
        self.gaze_point = None

    def set_fixations(self, fixations):
        self.fixations = fixations

    def set_current_aoi(self, name):
        self.current_aoi = name
        if name:
            self.header.setText(f"{self.participant_id.upper()} → {name}")
        else:
            self.header.setText(self.participant_id.upper())

    def set_scanpath(self, sp):
        self.scanpath = sp

    def set_aois(self, aois, colors, is_keyframe=None):
        self.aois = aois
        self.aoi_colors = colors
        self.aoi_is_keyframe = is_keyframe or {}
        self.update()

    def set_yolo_boxes(self, boxes):
        self.yolo_boxes = boxes
        self.update()

    def start_drawing_aoi(self):
        self.drawing_new_aoi = True
        self.setCursor(Qt.CursorShape.CrossCursor)

    def stop_drawing_aoi(self):
        self.drawing_new_aoi = False
        self.fastsam_mode = False
        self.draw_start = self.draw_current = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def _get_canvas_transform(self):
        if not self.image:
            return 0, 0, 1, 1
        cr = self.canvas.geometry()
        scaled = self.image.scaled(cr.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                   Qt.TransformationMode.SmoothTransformation)
        xo = cr.x() + (cr.width() - scaled.width()) // 2
        yo = cr.y() + (cr.height() - scaled.height()) // 2
        sx = scaled.width() / self.image.width()
        sy = scaled.height() / self.image.height()
        return xo, yo, sx, sy

    def _widget_to_frame(self, pos):
        xo, yo, sx, sy = self._get_canvas_transform()
        return (pos.x() - xo) / sx, (pos.y() - yo) / sy

    def _get_handle_at(self, pos, bbox) -> HandleType:
        fx, fy = self._widget_to_frame(pos)
        _, _, sx, _ = self._get_canvas_transform()
        hs = 12 / sx if sx > 0 else 12
        x1, y1, x2, y2 = bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height
        if abs(fx-x1)<hs and abs(fy-y1)<hs: return HandleType.TOP_LEFT
        if abs(fx-x2)<hs and abs(fy-y1)<hs: return HandleType.TOP_RIGHT
        if abs(fx-x1)<hs and abs(fy-y2)<hs: return HandleType.BOTTOM_LEFT
        if abs(fx-x2)<hs and abs(fy-y2)<hs: return HandleType.BOTTOM_RIGHT
        cx, cy = (x1+x2)/2, (y1+y2)/2
        if abs(fx-cx)<bbox.width/2-hs and abs(fy-y1)<hs: return HandleType.TOP
        if abs(fx-cx)<bbox.width/2-hs and abs(fy-y2)<hs: return HandleType.BOTTOM
        if abs(fx-x1)<hs and abs(fy-cy)<bbox.height/2-hs: return HandleType.LEFT
        if abs(fx-x2)<hs and abs(fy-cy)<bbox.height/2-hs: return HandleType.RIGHT
        if bbox.contains(fx, fy): return HandleType.MOVE
        return HandleType.NONE

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self._show_aoi_context_menu(event.pos(), event.globalPosition().toPoint())
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.pos()
        if self.fastsam_mode:
            fx, fy = self._widget_to_frame(pos)
            self.fastsam_click.emit(self.participant_id, self.current_frame_idx,
                                    float(fx), float(fy))
            return
        if self.drawing_new_aoi:
            self.draw_start = self.draw_current = pos
            return
        for name, bbox in self.aois.items():
            handle = self._get_handle_at(pos, bbox)
            if handle != HandleType.NONE:
                self.selected_aoi = name
                self.manipulating = True
                self.handle_type = handle
                self.manip_start = pos
                self.manip_original = bbox.copy()
                self.aoi_selected.emit(name)
                self.update()
                return
        self.selected_aoi = None
        self.update()

    def mouseMoveEvent(self, event):
        pos = event.pos()
        if self.drawing_new_aoi and self.draw_start:
            self.draw_current = pos
            self.update()
            return
        if self.manipulating and self.manip_original:
            dx = pos.x() - self.manip_start.x()
            dy = pos.y() - self.manip_start.y()
            _, _, sx, sy = self._get_canvas_transform()
            dxf = dx / sx if sx else 0
            dyf = dy / sy if sy else 0
            o = self.manip_original
            new = None
            if self.handle_type == HandleType.MOVE:
                new = BoundingBox(o.x+dxf, o.y+dyf, o.width, o.height)
            elif self.handle_type == HandleType.TOP_LEFT:
                new = BoundingBox(o.x+dxf, o.y+dyf, o.width-dxf, o.height-dyf)
            elif self.handle_type == HandleType.TOP_RIGHT:
                new = BoundingBox(o.x, o.y+dyf, o.width+dxf, o.height-dyf)
            elif self.handle_type == HandleType.BOTTOM_LEFT:
                new = BoundingBox(o.x+dxf, o.y, o.width-dxf, o.height+dyf)
            elif self.handle_type == HandleType.BOTTOM_RIGHT:
                new = BoundingBox(o.x, o.y, o.width+dxf, o.height+dyf)
            elif self.handle_type == HandleType.TOP:
                new = BoundingBox(o.x, o.y+dyf, o.width, o.height-dyf)
            elif self.handle_type == HandleType.BOTTOM:
                new = BoundingBox(o.x, o.y, o.width, o.height+dyf)
            elif self.handle_type == HandleType.LEFT:
                new = BoundingBox(o.x+dxf, o.y, o.width-dxf, o.height)
            elif self.handle_type == HandleType.RIGHT:
                new = BoundingBox(o.x, o.y, o.width+dxf, o.height)
            if new and abs(new.width) > 5 and abs(new.height) > 5:
                if new.width < 0: new.x += new.width; new.width = abs(new.width)
                if new.height < 0: new.y += new.height; new.height = abs(new.height)
                self.aois[self.selected_aoi] = new
                self.update()
            return
        # Hover cursor
        for name, bbox in self.aois.items():
            handle = self._get_handle_at(pos, bbox)
            if handle != HandleType.NONE:
                self.hovered_aoi = name
                cursors = {
                    HandleType.MOVE: Qt.CursorShape.SizeAllCursor,
                    HandleType.TOP_LEFT: Qt.CursorShape.SizeFDiagCursor,
                    HandleType.BOTTOM_RIGHT: Qt.CursorShape.SizeFDiagCursor,
                    HandleType.TOP_RIGHT: Qt.CursorShape.SizeBDiagCursor,
                    HandleType.BOTTOM_LEFT: Qt.CursorShape.SizeBDiagCursor,
                    HandleType.TOP: Qt.CursorShape.SizeVerCursor,
                    HandleType.BOTTOM: Qt.CursorShape.SizeVerCursor,
                }
                self.setCursor(cursors.get(handle, Qt.CursorShape.SizeHorCursor))
                self.update()
                return
        if not self.drawing_new_aoi and not self.fastsam_mode:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        if self.hovered_aoi:
            self.hovered_aoi = None
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self.drawing_new_aoi and self.draw_start and self.draw_current:
            fx1, fy1 = self._widget_to_frame(self.draw_start)
            fx2, fy2 = self._widget_to_frame(self.draw_current)
            x, y = min(fx1, fx2), min(fy1, fy2)
            w, h = abs(fx2-fx1), abs(fy2-fy1)
            if w > 10 and h > 10:
                # Offer preset names
                presets = self.aoi_presets + ["custom..."]
                name, ok = QInputDialog.getItem(self, "New AOI", "Select or type AOI name:",
                                                presets, 0, True)
                if ok and name:
                    if name == "custom...":
                        name, ok = QInputDialog.getText(self, "New AOI", "Enter AOI name:")
                    if ok and name:
                        self.aoi_created.emit(name, BoundingBox(x, y, w, h))
            self.stop_drawing_aoi()
            return
        if self.manipulating:
            if self.selected_aoi and self.selected_aoi in self.aois:
                self.aoi_updated.emit(self.selected_aoi, self.participant_id,
                                      self.current_frame_idx, self.aois[self.selected_aoi])
            self.manipulating = False
            self.handle_type = HandleType.NONE

    def _show_aoi_context_menu(self, pos, global_pos):
        """Right-click context menu on AOI boxes."""
        fx, fy = self._widget_to_frame(pos)
        clicked_aoi = None
        for name, bbox in self.aois.items():
            if bbox.contains(fx, fy):
                clicked_aoi = name
                break
        if not clicked_aoi:
            return
        self.selected_aoi = clicked_aoi
        self.aoi_selected.emit(clicked_aoi)
        self.update()

        menu = QMenu(self)
        menu.setStyleSheet(f"QMenu{{background-color:{Colors.BG_PANEL};color:white;"
                           f"border:1px solid {Colors.BORDER};}}"
                           f"QMenu::item:selected{{background-color:{Colors.ACCENT};}}")

        is_kf = self.aoi_is_keyframe.get(clicked_aoi, False)
        fidx = self.current_frame_idx

        if is_kf:
            del_kf = menu.addAction(f"🗑 Delete keyframe at frame {fidx}")
        else:
            del_kf = None
            menu.addAction(f"(no keyframe at frame {fidx})").setEnabled(False)

        menu.addSeparator()
        pin_kf = menu.addAction(f"📌 Pin current box as keyframe at frame {fidx}")

        action = menu.exec(global_pos)
        if action == del_kf and del_kf:
            self.aoi_keyframe_deleted.emit(clicked_aoi, self.participant_id, fidx)
        elif action == pin_kf:
            # Pin the current box as an actual keyframe (no change to position)
            bbox = self.aois[clicked_aoi]
            self.aoi_updated.emit(clicked_aoi, self.participant_id, fidx, bbox)

    def delete_selected_keyframe(self):
        """Delete keyframe at current frame for the selected AOI (called from keyboard shortcut)."""
        if self.selected_aoi and self.aoi_is_keyframe.get(self.selected_aoi, False):
            self.aoi_keyframe_deleted.emit(self.selected_aoi, self.participant_id,
                                           self.current_frame_idx)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        cr = self.canvas.geometry()
        painter.fillRect(cr, QColor(Colors.BG_DARK))
        if not self.image:
            painter.setPen(QColor(Colors.TEXT_DIM))
            painter.drawText(cr, Qt.AlignmentFlag.AlignCenter, "No video")
            return
        scaled = self.image.scaled(cr.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                   Qt.TransformationMode.SmoothTransformation)
        xo = cr.x() + (cr.width() - scaled.width()) // 2
        yo = cr.y() + (cr.height() - scaled.height()) // 2
        painter.drawPixmap(xo, yo, scaled)
        sx = scaled.width() / self.image.width()
        sy = scaled.height() / self.image.height()

        # YOLO boxes
        if self.show_yolo and self.yolo_boxes:
            for det in self.yolo_boxes:
                tid = det.get('track_id', -1)
                if tid in self.hidden_yolo_tracks:
                    continue
                bx, by, bw, bh = det['bbox']
                rx = xo + bx * sx; ry = yo + by * sy; rw = bw * sx; rh = bh * sy
                # Per-track color
                tc = self.TRACK_COLORS[tid % len(self.TRACK_COLORS)]
                is_hl = (self.highlighted_yolo_track is not None and
                         tid == self.highlighted_yolo_track)
                c = QColor(tc)
                # Highlighted track: thick solid line, others: thinner
                if is_hl:
                    c.setAlpha(220)
                    painter.setPen(QPen(c, 3, Qt.PenStyle.SolidLine))
                    fill = QColor(tc); fill.setAlpha(40)
                    painter.setBrush(fill)
                else:
                    c.setAlpha(160)
                    painter.setPen(QPen(c, 2, Qt.PenStyle.SolidLine))
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(int(rx), int(ry), int(rw), int(rh))
                # Label: big track ID + class
                cls = det.get('class', '?')
                conf = det.get('confidence', 0)
                label = f"T{tid} {cls} {conf:.0%}"
                font = painter.font(); font.setPointSize(11); font.setBold(True)
                painter.setFont(font)
                # Label background
                fm = painter.fontMetrics()
                tw = fm.horizontalAdvance(label) + 8
                th = fm.height() + 4
                bg = QColor(tc); bg.setAlpha(180)
                painter.setPen(Qt.PenStyle.NoPen); painter.setBrush(bg)
                painter.drawRect(int(rx), int(ry - th), tw, th)
                # Label text
                painter.setPen(QColor("#000000"))
                painter.drawText(int(rx + 4), int(ry - 4), label)
                font.setBold(False); painter.setFont(font)

        # AOIs
        if self.show_aois:
            for name, bbox in self.aois.items():
                color = QColor(self.aoi_colors.get(name, "#ffffff"))
                is_sel = name == self.selected_aoi
                is_cur = name == self.current_aoi
                is_oob = not bbox.is_in_bounds(self.frame_width, self.frame_height)
                bx = xo + bbox.x * sx; by = yo + bbox.y * sy
                bw = bbox.width * sx; bh = bbox.height * sy
                fill = QColor(color)
                fill.setAlpha(60 if is_cur else (40 if is_sel else (15 if is_oob else 20)))
                painter.fillRect(int(bx), int(by), int(bw), int(bh), fill)
                pw = 3 if (is_sel or is_cur) else 1
                ps = Qt.PenStyle.DashLine if is_oob else Qt.PenStyle.SolidLine
                painter.setPen(QPen(color, pw, ps))
                painter.drawRect(int(bx), int(by), int(bw), int(bh))
                lbl = name + (" (OOB)" if is_oob else "")
                kf_marker = " ◆" if self.aoi_is_keyframe.get(name, False) else " ◇"
                lbl += kf_marker
                painter.setPen(color)
                font = painter.font(); font.setBold(True); font.setPointSize(10); painter.setFont(font)
                painter.drawText(int(bx+4), int(by+14), lbl)
                if is_sel:
                    painter.setBrush(color); painter.setPen(Qt.PenStyle.NoPen)
                    for cx_, cy_ in [(bx,by),(bx+bw,by),(bx,by+bh),(bx+bw,by+bh)]:
                        painter.drawRect(int(cx_-4), int(cy_-4), 8, 8)

        # Scanpath
        if self.show_scanpath and len(self.scanpath) > 1:
            for i in range(1, len(self.scanpath)):
                prev, curr = self.scanpath[i-1], self.scanpath[i]
                px1 = xo + prev.x * sx; py1 = yo + prev.y * sy
                px2 = xo + curr.x * sx; py2 = yo + curr.y * sy
                if prev.aoi_name and curr.aoi_name and prev.aoi_name != curr.aoi_name:
                    painter.setPen(QPen(QColor("#ffd700"), 3))
                elif curr.aoi_name:
                    c = QColor(self.aoi_colors.get(curr.aoi_name, Colors.SCANPATH)); c.setAlpha(180)
                    painter.setPen(QPen(c, 2))
                else:
                    painter.setPen(QPen(QColor(Colors.SCANPATH), 1))
                painter.drawLine(int(px1), int(py1), int(px2), int(py2))

        # Trail
        if self.show_trail and len(self.gaze_trail) > 1:
            trail = list(self.gaze_trail)
            for i, (gx, gy) in enumerate(trail):
                a = int(50 + (i/len(trail))*150)
                sz = 2 + (i/len(trail))*4
                c = QColor(self.color); c.setAlpha(a)
                painter.setPen(Qt.PenStyle.NoPen); painter.setBrush(c)
                px = xo + gx * sx; py = yo + gy * sy
                painter.drawEllipse(int(px-sz/2), int(py-sz/2), int(sz), int(sz))

        # Fixations
        if self.show_fixations and self.fixations:
            for fx, fy, dur in self.fixations:
                r = int(6 + dur/40)
                c = QColor(Colors.FIXATION); c.setAlpha(140)
                px = xo + int(fx * sx); py = yo + int(fy * sy)
                painter.setPen(QPen(c, 2)); painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(px-r, py-r, r*2, r*2)

        # Gaze crosshair
        if self.show_gaze and self.gaze_point:
            gx, gy = self.gaze_point
            px = xo + int(gx * sx); py = yo + int(gy * sy)
            c = QColor(self.color)
            painter.setPen(QPen(c, 2))
            painter.drawLine(px-12, py, px-4, py); painter.drawLine(px+4, py, px+12, py)
            painter.drawLine(px, py-12, px, py-4); painter.drawLine(px, py+4, px, py+12)
            painter.setPen(Qt.PenStyle.NoPen); painter.setBrush(c)
            painter.drawEllipse(px-4, py-4, 8, 8)

        # Draw-in-progress rectangle
        if self.drawing_new_aoi and self.draw_start and self.draw_current:
            painter.setPen(QPen(QColor("#00ff00"), 2, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            x1, y1 = self.draw_start.x(), self.draw_start.y()
            x2, y2 = self.draw_current.x(), self.draw_current.y()
            painter.drawRect(min(x1,x2), min(y1,y2), abs(x2-x1), abs(y2-y1))
