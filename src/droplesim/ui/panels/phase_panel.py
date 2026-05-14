"""Phase region list panel with delete."""

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


class PhasePanel(QWidget):
    edit_requested = Signal(int)
    delete_requested = Signal(int)

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

        layout.addWidget(QLabel("Phase Regions"))
        self._list = QListWidget()
        layout.addWidget(self._list)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        edit_btn = ui.button("Edit")
        edit_btn.clicked.connect(self._on_edit)
        btn_row.addWidget(edit_btn)
        del_btn = ui.button("Delete", variant="danger")
        del_btn.clicked.connect(self._on_delete)
        btn_row.addWidget(del_btn)
        layout.addLayout(btn_row)

    def set_regions(self, regions: list[dict]):
        self._list.clear()
        for i, r in enumerate(regions):
            phi = r.get("phi", 1.0)
            if phi > 0.5:
                label = "oil"
                color = "#e74c3c"
            else:
                label = "aqueous"
                color = "#3498db"
            text = f"Region {i}: phi={phi:.1f} ({label})"
            item = QListWidgetItem(text)
            item.setForeground(QColor(color))
            self._list.addItem(item)

    def _on_edit(self):
        row = self._list.currentRow()
        if row >= 0:
            self.edit_requested.emit(row)

    def _on_delete(self):
        row = self._list.currentRow()
        if row >= 0:
            self.delete_requested.emit(row)
