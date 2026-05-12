"""Phase region list panel with delete."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class PhasePanel(QWidget):
    delete_requested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        layout.addWidget(QLabel("Phase Regions"))
        self._list = QListWidget()
        layout.addWidget(self._list)

        del_btn = QPushButton("Delete")
        del_btn.clicked.connect(self._on_delete)
        layout.addWidget(del_btn)

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

    def _on_delete(self):
        row = self._list.currentRow()
        if row >= 0:
            self.delete_requested.emit(row)
