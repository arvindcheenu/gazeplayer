"""Session timeline with playhead and event markers."""

from typing import Dict

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QLinearGradient

from .theme import Colors


class SyncTimeline(QWidget):
    seek_requested = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(80); self.setMaximumHeight(80)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.duration_s = 0; self.current_time_s = 0
        self.events = []; self.dragging = False; self.margin = 16
        self.participant_colors: Dict[str, str] = {}

    def set_duration(self, d): self.duration_s = d; self.update()
    def set_current_time(self, t): self.current_time_s = t; self.update()
    def set_events(self, e): self.events = e; self.update()
    def set_participant_colors(self, c): self.participant_colors = dict(c); self.update()

    def _time_to_x(self, t):
        if self.duration_s <= 0: return self.margin
        return self.margin + (t / self.duration_s) * (self.width() - self.margin*2)

    def _x_to_time(self, x):
        vw = self.width() - self.margin*2
        if vw <= 0: return 0
        return max(0, min(1, (x - self.margin) / vw)) * self.duration_s

    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(Colors.BG_PANEL))
        vw = self.width() - self.margin*2
        ty, th = 35, 20
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor("#0d0d0d"))
        p.drawRoundedRect(self.margin, ty, vw, th, 4, 4)
        if self.duration_s > 0:
            pw = (self.current_time_s / self.duration_s) * vw
            g = QLinearGradient(self.margin, 0, self.margin + pw, 0)
            g.setColorAt(0, QColor(Colors.ACCENT)); g.setColorAt(1, QColor("#0066cc"))
            p.setBrush(g); p.drawRoundedRect(self.margin, ty, int(pw), th, 4, 4)
        for t_s, name, pid in self.events:
            x = self._time_to_x(t_s)
            c = "#ffd700" if 'transition' in name else self.participant_colors.get(pid, Colors.EVENT)
            p.setPen(QPen(QColor(c), 1)); p.drawLine(int(x), ty, int(x), ty+th)
        phx = self._time_to_x(self.current_time_s)
        p.setPen(QPen(QColor(Colors.PLAYHEAD), 2))
        p.drawLine(int(phx), ty-10, int(phx), ty+th+10)
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(QColor(Colors.PLAYHEAD))
        p.drawRoundedRect(int(phx-6), ty+th+12, 12, 5, 2, 2)
        p.setPen(QColor(Colors.TEXT_DIM)); f = p.font(); f.setPointSize(10); p.setFont(f)
        cm, cs = int(self.current_time_s//60), self.current_time_s%60
        p.drawText(self.margin, 20, f"{cm}:{cs:05.2f}")
        if self.duration_s > 0:
            dm, ds = int(self.duration_s//60), self.duration_s%60
            p.drawText(self.width()-self.margin-60, 20, f"{dm}:{ds:05.2f}")

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.dragging = True; self._seek(e.pos().x())
    def mouseMoveEvent(self, e):
        if self.dragging: self._seek(e.pos().x())
    def mouseReleaseEvent(self, e): self.dragging = False
    def _seek(self, x):
        t = self._x_to_time(x)
        if t != self.current_time_s:
            self.current_time_s = t; self.seek_requested.emit(t); self.update()
