"""Reusable 2D field visualization: ImageItem + colormap LUT + ColorBar legend."""

from __future__ import annotations

import dropletui as ui
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt
from PySide6.QtWidgets import QVBoxLayout, QWidget

_TEXT_MUTED = ui.Theme.TEXT_MUTED
_TEXT_WHITE = ui.Theme.TEXT_WHITE


def _build_lut(cmap: pg.ColorMap) -> np.ndarray:
    """Build a (256, 3) uint8 RGB lookup table from a pyqtgraph ColorMap."""
    table = cmap.getLookupTable(nPts=256, alpha=False)
    return np.asarray(table, dtype=np.uint8)[:, :3]


class FieldPlot(QWidget):
    """Reusable 2D field visualization: ImageItem + colormap LUT + color bar."""

    def __init__(self, title: str, colormap: pg.ColorMap, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._title = title
        self._cmap = colormap
        self._lut = _build_lut(colormap)

        self._plot = pg.PlotWidget(title=title)
        self._plot.setBackground(ui.Theme.BG_DARK)
        self._plot.setAspectLocked(True)
        self._plot.setLabel("bottom", "x [µm]")
        self._plot.setLabel("left", "y [µm]")

        self._img = pg.ImageItem()
        self._plot.addItem(self._img)

        self._bar = pg.ColorBarItem(
            values=(0, 1), colorMap=colormap, interactive=False, width=15,
        )
        self._plot.plotItem.layout.addItem(self._bar, 2, 5)

        # Crosshair
        pen = pg.mkPen(color=_TEXT_MUTED, width=1, style=Qt.DotLine)
        self._vline = pg.InfiniteLine(angle=90, pen=pen, movable=False)
        self._hline = pg.InfiniteLine(angle=0, pen=pen, movable=False)
        self._vline.setVisible(False)
        self._hline.setVisible(False)
        self._plot.addItem(self._vline, ignoreBounds=True)
        self._plot.addItem(self._hline, ignoreBounds=True)

        self._cursor_label = pg.TextItem(anchor=(0, 1), color=_TEXT_WHITE)
        self._cursor_label.setVisible(False)
        self._plot.addItem(self._cursor_label, ignoreBounds=True)

        # Connect mouse events
        self._plot.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self._plot.scene().sigMouseClicked.connect(self._on_mouse_clicked)

        layout.addWidget(self._plot)

        self._dx_um = 2.5
        self._origin_um = (0.0, 0.0)
        self._ny = 0
        self._nx = 0
        self._fluid_y = None
        self._fluid_x = None
        self._rgba = None
        self._index_map = None
        self._data = None
        self._vmin = 0.0
        self._vmax = 1.0
        self._fmt = ".2e"

        # Probes: list of (ScatterPlotItem, TextItem, row, col)
        self._probes: list[tuple[pg.ScatterPlotItem, pg.TextItem, int, int]] = []

    def set_geometry(self, ny: int, nx: int, fluid_y, fluid_x, dx_um: float,
                     origin_um: tuple[float, float],
                     index_map: np.ndarray | None = None):
        self._ny = ny
        self._nx = nx
        self._fluid_y = fluid_y
        self._fluid_x = fluid_x
        self._dx_um = dx_um
        self._origin_um = origin_um
        self._index_map = index_map

        self._rgba = np.zeros((ny, nx, 4), dtype=np.uint8)
        self._rgba[fluid_y, fluid_x, 3] = 255
        self.clear_probes()

    def clear_geometry(self):
        self._fluid_y = None
        self._fluid_x = None
        self._rgba = None
        self._index_map = None
        self._data = None
        self.clear_probes()

    def update(self, data: np.ndarray, vmin: float | None = None,
               vmax: float | None = None, fmt: str = ".2e"):
        if self._rgba is None or self._fluid_y is None:
            return
        if vmin is None:
            vmin = float(data.min())
        if vmax is None:
            vmax = float(data.max())
        span = vmax - vmin
        if span == 0:
            span = 1.0

        self._data = data
        self._vmin = vmin
        self._vmax = vmax
        self._fmt = fmt

        idx = np.clip(((data - vmin) / span * 255).astype(np.uint8), 0, 255)
        self._rgba[self._fluid_y, self._fluid_x, :3] = self._lut[idx]

        ox, oy = self._origin_um
        dx = self._dx_um
        rect = QRectF(ox, oy, self._nx * dx, self._ny * dx)
        self._img.setImage(self._rgba.transpose(1, 0, 2))
        self._img.setRect(rect)

        self._bar.setLevels(values=(vmin, vmax))
        self._refresh_probes()

    def auto_range(self):
        self._plot.autoRange()

    @property
    def plot_widget(self) -> pg.PlotWidget:
        return self._plot

    # ── Probing ──────────────────────────────────────────────────────────

    def _value_at(self, x_um: float, y_um: float) -> float | None:
        if self._index_map is None or self._data is None:
            return None
        ox, oy = self._origin_um
        dx = self._dx_um
        col = int((x_um - ox) / dx)
        row = int((y_um - oy) / dx)
        if 0 <= row < self._ny and 0 <= col < self._nx:
            si = int(self._index_map[row, col])
            if si >= 0:
                return float(self._data[si])
        return None

    def _grid_rc(self, x_um: float, y_um: float) -> tuple[int, int] | None:
        ox, oy = self._origin_um
        dx = self._dx_um
        col = int((x_um - ox) / dx)
        row = int((y_um - oy) / dx)
        if 0 <= row < self._ny and 0 <= col < self._nx:
            return row, col
        return None

    def _on_mouse_moved(self, pos):
        vb = self._plot.plotItem.vb
        if not vb.sceneBoundingRect().contains(pos):
            self._vline.setVisible(False)
            self._hline.setVisible(False)
            self._cursor_label.setVisible(False)
            return
        mp = vb.mapSceneToView(pos)
        x_um, y_um = mp.x(), mp.y()
        self._vline.setPos(x_um)
        self._hline.setPos(y_um)
        self._vline.setVisible(True)
        self._hline.setVisible(True)

        val = self._value_at(x_um, y_um)
        if val is not None:
            txt = f"x={x_um:.1f}  y={y_um:.1f}  val={val:{self._fmt}}"
        else:
            txt = f"x={x_um:.1f}  y={y_um:.1f}"
        self._cursor_label.setText(txt)
        self._cursor_label.setPos(x_um, y_um)
        self._cursor_label.setVisible(True)

    def _on_mouse_clicked(self, event):
        if event.button() not in (Qt.LeftButton, Qt.MouseButton.LeftButton):
            return
        pos = event.scenePos()
        vb = self._plot.plotItem.vb
        if not vb.sceneBoundingRect().contains(pos):
            return
        mp = vb.mapSceneToView(pos)
        x_um, y_um = mp.x(), mp.y()
        rc = self._grid_rc(x_um, y_um)
        if rc is None:
            return

        if event.double():
            self._remove_nearest_probe(x_um, y_um)
        else:
            row, col = rc
            if self._index_map is not None and self._index_map[row, col] >= 0:
                self._add_probe(row, col)

    def _add_probe(self, row: int, col: int):
        ox, oy = self._origin_um
        dx = self._dx_um
        px = ox + (col + 0.5) * dx
        py = oy + (row + 0.5) * dx

        marker = pg.ScatterPlotItem(
            [px], [py], symbol="+", size=12,
            pen=pg.mkPen(_TEXT_WHITE, width=2), brush=None,
        )
        label = pg.TextItem(anchor=(0, 1), color=_TEXT_WHITE)
        label.setPos(px, py)
        self._plot.addItem(marker)
        self._plot.addItem(label, ignoreBounds=True)

        self._probes.append((marker, label, row, col))
        self._refresh_probe_label(len(self._probes) - 1)

    def _remove_nearest_probe(self, x_um: float, y_um: float):
        if not self._probes:
            return
        ox, oy = self._origin_um
        dx = self._dx_um
        best_i, best_d = 0, float("inf")
        for i, (_, _, row, col) in enumerate(self._probes):
            px = ox + (col + 0.5) * dx
            py = oy + (row + 0.5) * dx
            d = (px - x_um) ** 2 + (py - y_um) ** 2
            if d < best_d:
                best_i, best_d = i, d
        marker, label, _, _ = self._probes.pop(best_i)
        self._plot.removeItem(marker)
        self._plot.removeItem(label)

    def _refresh_probe_label(self, idx: int):
        _, label, row, col = self._probes[idx]
        if self._index_map is not None and self._data is not None:
            si = int(self._index_map[row, col])
            if si >= 0:
                val = float(self._data[si])
                label.setText(f"{val:{self._fmt}}")
                return
        label.setText("--")

    def _refresh_probes(self):
        for i in range(len(self._probes)):
            self._refresh_probe_label(i)

    def clear_probes(self):
        for marker, label, _, _ in self._probes:
            self._plot.removeItem(marker)
            self._plot.removeItem(label)
        self._probes.clear()
