"""Tab 1: Geometry — solid mask display with µm axes."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QVBoxLayout, QWidget


class GeometryView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._plot = pg.PlotWidget(title="Solid Mask")
        self._plot.setBackground("#1a1a1a")
        self._plot.setAspectLocked(True)
        self._plot.setLabel("bottom", "x [µm]")
        self._plot.setLabel("left", "y [µm]")
        layout.addWidget(self._plot)

        self._img = pg.ImageItem()
        self._plot.addItem(self._img)

        lut = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            lut[i] = [i, i, i]
        self._img.setLookupTable(lut)

    def set_geometry(self, solid_mask: np.ndarray, dx_um: float, origin_um: tuple[float, float]):
        ny, nx = solid_mask.shape
        ox, oy = origin_um
        display = (~solid_mask).astype(np.float32)
        self._img.setImage(display.T, levels=[0, 1])
        self._img.setRect(ox, oy, nx * dx_um, ny * dx_um)
        self._plot.autoRange()

    def clear(self):
        self._img.clear()
