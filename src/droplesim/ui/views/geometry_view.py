"""Tab 1: Geometry — solid mask display with µm axes."""

from __future__ import annotations

import dropletui as ui
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QLabel, QListWidget, QListWidgetItem, QVBoxLayout, QWidget


class GeometryView(QWidget):
    solid_mask_changed = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._solid_mask = None
        self._dx_um = 2.5
        self._origin_um = (0.0, 0.0)
        self._labels = None
        self._areas: list[dict] = []
        self._selected_area = -1

        self._plot = pg.PlotWidget(title="Solid Mask")
        self._plot.setBackground("#1a1a1a")
        self._plot.setAspectLocked(True)
        self._plot.setLabel("bottom", "x [µm]")
        self._plot.setLabel("left", "y [µm]")
        self._plot.scene().sigMouseClicked.connect(self._on_plot_click)

        self._img = pg.ImageItem()
        self._plot.addItem(self._img)
        self._selection_item = pg.ImageItem()
        self._plot.addItem(self._selection_item)

        lut = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            lut[i] = [i, i, i]
        self._img.setLookupTable(lut)

        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(
            ui.Theme.PANEL_PADDING,
            ui.Theme.PANEL_PADDING,
            ui.Theme.PANEL_PADDING,
            ui.Theme.PANEL_PADDING,
        )
        panel_layout.setSpacing(ui.Theme.SPACE_2)
        panel_layout.addWidget(QLabel("White Areas"))
        self._area_list = QListWidget()
        self._area_list.currentRowChanged.connect(self._select_area)
        panel_layout.addWidget(self._area_list)
        self._deselect_btn = ui.button("Deselect", variant="danger")
        self._deselect_btn.clicked.connect(self._deselect_selected_area)
        panel_layout.addWidget(self._deselect_btn)

        layout.addWidget(
            ui.split_view(
                self._plot,
                panel,
                side_position="right",
                sizes=(1000, 320),
            )
        )

    def set_geometry(self, solid_mask: np.ndarray, dx_um: float, origin_um: tuple[float, float]):
        self._solid_mask = solid_mask.copy()
        self._dx_um = dx_um
        self._origin_um = origin_um
        self._selected_area = -1
        self._rebuild_areas()
        self._render()
        self._plot.autoRange()

    def _render(self):
        if self._solid_mask is None:
            return
        solid_mask = self._solid_mask
        dx_um = self._dx_um
        origin_um = self._origin_um
        ny, nx = solid_mask.shape
        ox, oy = origin_um
        display = (~solid_mask).astype(np.float32)
        self._img.setImage(display.T, levels=[0, 1])
        self._img.setRect(ox, oy, nx * dx_um, ny * dx_um)
        self._selection_item.setRect(QRectF(ox, oy, nx * dx_um, ny * dx_um))
        self._render_selection()

    def _rebuild_areas(self):
        self._areas = []
        self._labels = None
        self._area_list.clear()
        if self._solid_mask is None:
            return
        from scipy import ndimage as ndi

        fluid = ~self._solid_mask
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
        self._labels, nlab = ndi.label(fluid, structure=structure)
        for label_id in range(1, nlab + 1):
            ys, xs = np.where(self._labels == label_id)
            if len(xs) == 0:
                continue
            area = {
                "label_id": label_id,
                "cells": int(len(xs)),
                "x1": int(xs.min()),
                "x2": int(xs.max()),
                "y1": int(ys.min()),
                "y2": int(ys.max()),
            }
            self._areas.append(area)
            text = f"Area {len(self._areas) - 1}: {area['cells']:,} cells"
            item = QListWidgetItem(text)
            item.setForeground(QColor("#dddddd"))
            self._area_list.addItem(item)

    def _select_area(self, row: int):
        self._selected_area = row if 0 <= row < len(self._areas) else -1
        self._render_selection()

    def _render_selection(self):
        if self._solid_mask is None or self._labels is None or self._selected_area < 0:
            self._selection_item.clear()
            return
        area = self._areas[self._selected_area]
        mask = self._labels == area["label_id"]
        rgba = np.zeros((*self._solid_mask.shape, 4), dtype=np.uint8)
        rgba[mask] = np.array([243, 156, 18, 130], dtype=np.uint8)
        self._selection_item.setImage(rgba.transpose(1, 0, 2))

    def _on_plot_click(self, ev):
        if self._solid_mask is None or self._labels is None:
            return
        vb = self._plot.getViewBox()
        pt = vb.mapSceneToView(ev.scenePos())
        mx, my = pt.x(), pt.y()
        ny, nx = self._solid_mask.shape
        ox, oy = self._origin_um
        ix = int((mx - ox) / self._dx_um)
        iy = int((my - oy) / self._dx_um)
        if not (0 <= ix < nx and 0 <= iy < ny):
            return
        label_id = self._labels[iy, ix]
        if label_id == 0:
            return
        for idx, area in enumerate(self._areas):
            if area["label_id"] == label_id:
                self._area_list.setCurrentRow(idx)
                return

    def _deselect_selected_area(self):
        if self._solid_mask is None or self._labels is None or self._selected_area < 0:
            return
        area = self._areas[self._selected_area]
        self._solid_mask[self._labels == area["label_id"]] = True
        self._selected_area = -1
        self._rebuild_areas()
        self._render()
        self.solid_mask_changed.emit(self._solid_mask.copy())

    def get_areas(self) -> list[dict]:
        return [dict(area) for area in self._areas]

    def clear(self):
        self._solid_mask = None
        self._labels = None
        self._areas = []
        self._area_list.clear()
        self._selection_item.clear()
        self._img.clear()
