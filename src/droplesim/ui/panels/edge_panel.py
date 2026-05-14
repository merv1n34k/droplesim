"""Edge + area BC list panel with edit/delete buttons."""

from __future__ import annotations

import dropletui as ui
from PySide6.QtCore import Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

_KIND_COLORS = {
    "wall": "#888888",
    "inlet": "#3498db",
    "outlet": "#e74c3c",
}


class EdgePanel(QWidget):
    edit_edge_requested = Signal(int)
    delete_edge_requested = Signal(int)
    edit_area_requested = Signal(int)
    delete_area_requested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            ui.Theme.PANEL_PADDING,
            ui.Theme.PANEL_PADDING,
            ui.Theme.PANEL_PADDING,
            ui.Theme.PANEL_PADDING,
        )
        layout.setSpacing(ui.Theme.SPACE_2)

        # -- Edge list --
        layout.addWidget(QLabel("Edges"))
        self._edge_list = QListWidget()
        layout.addWidget(self._edge_list)

        edge_btn_row = QHBoxLayout()
        edge_btn_row.setSpacing(4)
        edit_btn = ui.button("Edit")
        edit_btn.clicked.connect(self._on_edit_edge)
        edge_btn_row.addWidget(edit_btn)
        reset_btn = ui.button("Reset", variant="danger")
        reset_btn.clicked.connect(self._on_delete_edge)
        edge_btn_row.addWidget(reset_btn)
        layout.addLayout(edge_btn_row)

        # -- Area BC list --
        layout.addWidget(QLabel("Area BCs"))
        self._area_list = QListWidget()
        layout.addWidget(self._area_list)

        area_btn_row = QHBoxLayout()
        area_btn_row.setSpacing(4)
        area_edit_btn = ui.button("Edit")
        area_edit_btn.clicked.connect(self._on_edit_area)
        area_btn_row.addWidget(area_edit_btn)
        area_del_btn = ui.button("Delete", variant="danger")
        area_del_btn.clicked.connect(self._on_delete_area)
        area_btn_row.addWidget(area_del_btn)
        layout.addLayout(area_btn_row)

    def set_edges(self, edges: list[dict]):
        self._edge_list.clear()
        for i, e in enumerate(edges):
            kind = e.get("kind", "wall")
            name = e.get("name", f"edge_{i}")
            color = _KIND_COLORS.get(kind, "#888888")
            item = QListWidgetItem(f"[{kind}] {name}")
            item.setForeground(QColor(color))
            self._edge_list.addItem(item)

    def set_areas(self, areas: list[dict]):
        self._area_list.clear()
        for a in areas:
            kind = a.get("kind", "inlet")
            name = a.get("name", "area")
            color = _KIND_COLORS.get(kind, "#888888")
            item = QListWidgetItem(f"[{kind}] {name}")
            item.setForeground(QColor(color))
            self._area_list.addItem(item)

    def _on_edit_edge(self):
        row = self._edge_list.currentRow()
        if row >= 0:
            self.edit_edge_requested.emit(row)

    def _on_delete_edge(self):
        row = self._edge_list.currentRow()
        if row >= 0:
            self.delete_edge_requested.emit(row)

    def _on_edit_area(self):
        row = self._area_list.currentRow()
        if row >= 0:
            self.edit_area_requested.emit(row)

    def _on_delete_area(self):
        row = self._area_list.currentRow()
        if row >= 0:
            self.delete_area_requested.emit(row)
