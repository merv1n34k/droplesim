"""Tab 3: Rectangle selection → fluid-only phase fill."""

from __future__ import annotations

import logging

import dropletui as ui
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QVBoxLayout,
    QWidget,
)

from droplesim.ui.panels.phase_panel import PhasePanel

log = logging.getLogger(__name__)

_WALL_COLOR = (150, 150, 150, 220)
# RGBA for phase fill overlays (only fluid cells are colored)
_OIL_RGBA = np.array([231, 76, 60, 100], dtype=np.uint8)
_AQ_RGBA = np.array([52, 152, 219, 100], dtype=np.uint8)
_OVERLAP_RGBA = np.array([240, 240, 240, 130], dtype=np.uint8)


class PhiDialog(QDialog):
    def __init__(self, parent=None, phi: float = 1.0):
        super().__init__(parent)
        self.setWindowTitle("Set Phase")
        self.setMinimumWidth(250)
        layout = QFormLayout(self)

        self._preset = ui.combo_box(["Oil (phi=1)", "Aqueous (phi=0)"])
        self._preset.currentIndexChanged.connect(self._on_preset)
        layout.addRow("Phase:", self._preset)

        self._phi = ui.double_box(
            minimum=0.0,
            maximum=1.0,
            value=1.0,
            step=0.1,
            decimals=2,
        )
        layout.addRow("Phi:", self._phi)
        self._preset.setCurrentIndex(0 if phi > 0.5 else 1)
        self._phi.setValue(phi)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def _on_preset(self, idx):
        self._phi.setValue(1.0 if idx == 0 else 0.0)

    def phi_value(self) -> float:
        return self._phi.value()


_DRAG_THRESHOLD = 3.0


class DragViewBox(pg.ViewBox):
    rect_drawn = Signal(float, float, float, float)
    point_clicked = Signal(float, float)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._press_pos = None
        self._drag_start = None
        self._dragging = False
        self._drag_rect = None
        self._bounds = None

    def set_bounds(self, xmin: float, ymin: float, xmax: float, ymax: float):
        self._bounds = (xmin, ymin, xmax, ymax)

    def _clamp(self, pt):
        x, y = pt.x(), pt.y()
        if self._bounds:
            x = max(self._bounds[0], min(x, self._bounds[2]))
            y = max(self._bounds[1], min(y, self._bounds[3]))
        return x, y

    def mousePressEvent(self, ev):
        if ev.button() == ev.button().LeftButton:
            self._press_pos = ev.pos()
            self._drag_start = self._clamp(self.mapToView(ev.pos()))
            self._dragging = False
            ev.accept()
        else:
            super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._press_pos is not None:
            delta = ev.pos() - self._press_pos
            dist = (delta.x() ** 2 + delta.y() ** 2) ** 0.5
            if dist > _DRAG_THRESHOLD:
                self._dragging = True
        if self._drag_start is not None and self._dragging:
            cx, cy = self._clamp(self.mapToView(ev.pos()))
            if self._drag_rect is not None:
                self.removeItem(self._drag_rect)
            x1, y1 = self._drag_start
            rect = QRectF(min(x1, cx), min(y1, cy), abs(cx - x1), abs(cy - y1))
            self._drag_rect = pg.QtWidgets.QGraphicsRectItem(rect)
            self._drag_rect.setPen(pg.mkPen("#f39c12", width=2, style=pg.QtCore.Qt.DashLine))
            self.addItem(self._drag_rect)
            ev.accept()
        else:
            super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._drag_start is not None and ev.button() == ev.button().LeftButton:
            if self._drag_rect is not None:
                self.removeItem(self._drag_rect)
                self._drag_rect = None
            if self._dragging:
                x2, y2 = self._clamp(self.mapToView(ev.pos()))
                x1, y1 = self._drag_start
                if abs(x2 - x1) > 1 and abs(y2 - y1) > 1:
                    self.rect_drawn.emit(
                        min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
                    )
            else:
                vpt = self.mapToView(ev.pos())
                self.point_clicked.emit(vpt.x(), vpt.y())
            self._press_pos = None
            self._drag_start = None
            self._dragging = False
            ev.accept()
        else:
            super().mouseReleaseEvent(ev)


class PhaseView(QWidget):
    phase_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._vb = DragViewBox()
        self._vb.rect_drawn.connect(self._on_rect_drawn)
        self._vb.point_clicked.connect(self._on_point_clicked)
        self._plot = pg.PlotWidget(
            viewBox=self._vb, title="Phase Regions  (drag rectangle to paint)"
        )
        self._plot.setBackground(ui.Theme.BG_DARK)
        self._plot.setAspectLocked(True)
        self._plot.setLabel("bottom", "x [µm]")
        self._plot.setLabel("left", "y [µm]")

        # Color bar: phi scale (blue=0/aqueous → white=0.5 → red=1/oil)
        phi_cmap = pg.ColorMap(
            pos=[0.0, 0.5, 1.0],
            color=[(52, 152, 219), (240, 240, 240), (231, 76, 60)],
        )
        self._phi_bar = pg.ColorBarItem(
            values=(0, 1), colorMap=phi_cmap, interactive=False, width=15,
            label="phi",
        )
        self._plot.plotItem.layout.addItem(self._phi_bar, 2, 5)

        self._panel = PhasePanel()
        self._panel.edit_requested.connect(self._on_edit_region)
        self._panel.delete_requested.connect(self._on_delete_region)
        layout.addWidget(
            ui.split_view(
                self._plot,
                self._panel,
                side_position="right",
                sizes=(1000, 320),
            )
        )

        self._wall_curves: list[pg.PlotDataItem] = []
        self._fill_items: list[pg.ImageItem] = []
        self._overlap_item: pg.ImageItem | None = None
        self._regions: list[dict] = []
        self._solid_mask = None
        self._dx_um = 2.5
        self._origin_um = (0.0, 0.0)

    def set_geometry(
        self,
        solid_mask: np.ndarray,
        dx_um: float,
        origin_um: tuple[float, float],
        edge_polylines_mm: list[list[tuple[float, float]]] | None = None,
    ):
        self._solid_mask = solid_mask
        self._dx_um = dx_um
        self._origin_um = origin_um

        ny, nx = solid_mask.shape
        ox, oy = origin_um
        self._vb.set_bounds(ox, oy, ox + nx * dx_um, oy + ny * dx_um)

        for c in self._wall_curves:
            self._plot.removeItem(c)
        self._wall_curves.clear()

        if edge_polylines_mm:
            for poly_mm in edge_polylines_mm:
                xs = [x * 1000.0 for x, y in poly_mm]
                ys = [y * 1000.0 for x, y in poly_mm]
                curve = self._plot.plot(
                    xs, ys, pen=pg.mkPen(color=_WALL_COLOR, width=1.5)
                )
                self._wall_curves.append(curve)

        self._redraw_regions()
        self._plot.autoRange()

    def _fluid_mask_for_region(self, region: dict) -> np.ndarray:
        """Build RGBA image showing only fluid cells inside the region."""
        ny, nx = self._solid_mask.shape
        ox, oy = self._origin_um
        dx = self._dx_um

        xs = ox + (np.arange(nx) + 0.5) * dx
        ys = oy + (np.arange(ny) + 0.5) * dx
        XX, YY = np.meshgrid(xs, ys)

        fluid_in_rect = (
            (XX >= region["x1_um"]) & (XX <= region["x2_um"])
            & (YY >= region["y1_um"]) & (YY <= region["y2_um"])
            & (~self._solid_mask)
        )

        rgba = np.zeros((ny, nx, 4), dtype=np.uint8)
        color = _OIL_RGBA if region["phi"] > 0.5 else _AQ_RGBA
        rgba[fluid_in_rect] = color
        return rgba

    def _redraw_overlap(self):
        """Show white overlay where oil and aqueous regions overlap (phi=0.5)."""
        if self._overlap_item is not None:
            self._plot.removeItem(self._overlap_item)
            self._overlap_item = None
        if self._solid_mask is None or not self._regions:
            return
        ny, nx = self._solid_mask.shape
        ox, oy = self._origin_um
        dx = self._dx_um
        xs = ox + (np.arange(nx) + 0.5) * dx
        ys = oy + (np.arange(ny) + 0.5) * dx
        XX, YY = np.meshgrid(xs, ys)

        has_oil = np.zeros((ny, nx), dtype=bool)
        has_aq = np.zeros((ny, nx), dtype=bool)
        for r in self._regions:
            in_rect = (
                (XX >= r["x1_um"]) & (XX <= r["x2_um"])
                & (YY >= r["y1_um"]) & (YY <= r["y2_um"])
                & (~self._solid_mask)
            )
            if r["phi"] > 0.5:
                has_oil |= in_rect
            else:
                has_aq |= in_rect

        overlap = has_oil & has_aq
        if not overlap.any():
            return
        rgba = np.zeros((ny, nx, 4), dtype=np.uint8)
        rgba[overlap] = _OVERLAP_RGBA
        self._overlap_item = pg.ImageItem()
        self._overlap_item.setImage(rgba.transpose(1, 0, 2))
        self._overlap_item.setRect(QRectF(ox, oy, nx * dx, ny * dx))
        self._plot.addItem(self._overlap_item)

    def _on_rect_drawn(self, x1, y1, x2, y2):
        if self._solid_mask is None:
            return
        dlg = PhiDialog(self)
        if dlg.exec():
            phi = dlg.phi_value()
            region = {
                "x1_um": x1, "y1_um": y1,
                "x2_um": x2, "y2_um": y2,
                "phi": phi,
            }
            self._regions.append(region)
            self._add_fill_item(region)
            self._redraw_overlap()
            self._panel.set_regions(self._regions)
            self.phase_changed.emit()
            label = "oil" if phi > 0.5 else "aqueous"
            log.info("Phase region added: %s (phi=%.1f)", label, phi)

    def _on_point_clicked(self, mx: float, my: float):
        for i in range(len(self._regions) - 1, -1, -1):
            r = self._regions[i]
            if (
                r["x1_um"] <= mx <= r["x2_um"]
                and r["y1_um"] <= my <= r["y2_um"]
            ):
                self._on_edit_region(i)
                return

    def _add_fill_item(self, region: dict):
        rgba = self._fluid_mask_for_region(region)
        ny, nx = self._solid_mask.shape
        ox, oy = self._origin_um
        dx = self._dx_um

        img = pg.ImageItem()
        # ImageItem expects (width, height, 4) so transpose first two axes
        img.setImage(rgba.transpose(1, 0, 2))
        img.setRect(QRectF(ox, oy, nx * dx, ny * dx))
        self._plot.addItem(img)
        self._fill_items.append(img)

    def _redraw_regions(self):
        for item in self._fill_items:
            self._plot.removeItem(item)
        self._fill_items.clear()
        for r in self._regions:
            self._add_fill_item(r)
        self._redraw_overlap()
        self._panel.set_regions(self._regions)

    def _on_edit_region(self, idx: int):
        if not (0 <= idx < len(self._regions)):
            return
        region = self._regions[idx]
        dlg = PhiDialog(self, phi=region.get("phi", 1.0))
        if dlg.exec():
            region["phi"] = dlg.phi_value()
            self._redraw_regions()
            self.phase_changed.emit()

    def _on_delete_region(self, idx: int):
        if 0 <= idx < len(self._regions):
            self._regions.pop(idx)
            self._redraw_regions()
            self.phase_changed.emit()

    def get_regions(self) -> list[dict]:
        return self._regions

    def build_phi_init(self) -> np.ndarray | None:
        if not self._regions or self._solid_mask is None:
            return None
        ny, nx = self._solid_mask.shape
        ox, oy = self._origin_um
        phi = np.ones((ny, nx), dtype=np.float64)

        xs = ox + (np.arange(nx) + 0.5) * self._dx_um
        ys = oy + (np.arange(ny) + 0.5) * self._dx_um
        XX, YY = np.meshgrid(xs, ys)

        has_oil = np.zeros((ny, nx), dtype=bool)
        has_aq = np.zeros((ny, nx), dtype=bool)
        for r in self._regions:
            in_rect = (
                (XX >= r["x1_um"]) & (XX <= r["x2_um"])
                & (YY >= r["y1_um"]) & (YY <= r["y2_um"])
                & (~self._solid_mask)
            )
            if r["phi"] > 0.5:
                has_oil |= in_rect
            else:
                has_aq |= in_rect

        phi[has_aq & ~has_oil] = 0.0
        phi[has_oil & has_aq] = 0.5

        return phi

    def set_regions_from_state(self, regions: list[dict]):
        self._regions = list(regions)
        self._redraw_regions()
