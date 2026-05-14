"""Reusable 2D field visualization: ImageItem + colormap LUT + min/max legend."""

from __future__ import annotations

import dropletui as ui
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF
from PySide6.QtWidgets import QVBoxLayout, QWidget


def _build_lut(cmap: pg.ColorMap) -> np.ndarray:
    """Build a (256, 3) uint8 RGB lookup table from a pyqtgraph ColorMap."""
    table = cmap.getLookupTable(nPts=256, alpha=False)
    return np.asarray(table, dtype=np.uint8)[:, :3]


class FieldPlot(QWidget):
    """Reusable 2D field visualization: ImageItem + colormap LUT + legend."""

    def __init__(self, title: str, colormap: pg.ColorMap, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._title = title
        self._lut = _build_lut(colormap)

        self._plot = pg.PlotWidget(title=title)
        self._plot.setBackground(ui.Theme.BG_DARK)
        self._plot.setAspectLocked(True)
        self._plot.setLabel("bottom", "x [µm]")
        self._plot.setLabel("left", "y [µm]")

        self._img = pg.ImageItem()
        self._plot.addItem(self._img)

        self._legend = pg.TextItem(anchor=(1, 0))
        self._legend.setColor("#cccccc")
        font = self._legend.textItem.font()
        font.setPointSize(9)
        self._legend.setFont(font)
        self._plot.addItem(self._legend)
        self._legend.setVisible(False)

        layout.addWidget(self._plot)

        self._dx_um = 2.5
        self._origin_um = (0.0, 0.0)
        self._ny = 0
        self._nx = 0
        self._fluid_y = None
        self._fluid_x = None
        self._rgba = None

    def set_geometry(self, ny: int, nx: int, fluid_y, fluid_x, dx_um: float,
                     origin_um: tuple[float, float]):
        self._ny = ny
        self._nx = nx
        self._fluid_y = fluid_y
        self._fluid_x = fluid_x
        self._dx_um = dx_um
        self._origin_um = origin_um

        self._rgba = np.zeros((ny, nx, 4), dtype=np.uint8)
        self._rgba[fluid_y, fluid_x, 3] = 255

    def clear_geometry(self):
        self._fluid_y = None
        self._fluid_x = None
        self._rgba = None

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

        idx = np.clip(((data - vmin) / span * 255).astype(np.uint8), 0, 255)
        self._rgba[self._fluid_y, self._fluid_x, :3] = self._lut[idx]

        ox, oy = self._origin_um
        dx = self._dx_um
        rect = QRectF(ox, oy, self._nx * dx, self._ny * dx)
        self._img.setImage(self._rgba.transpose(1, 0, 2))
        self._img.setRect(rect)

        self._legend.setText(f"{self._title}: {vmin:{fmt}} – {vmax:{fmt}}")
        vr = self._plot.viewRange()
        self._legend.setPos(vr[0][1], vr[1][1])
        self._legend.setVisible(True)

    def auto_range(self):
        self._plot.autoRange()

    @property
    def plot_widget(self) -> pg.PlotWidget:
        return self._plot
