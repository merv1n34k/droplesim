"""Edge list panel with edit/delete buttons."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

_KIND_COLORS = {
    "wall": "#888888",
    "inlet": "#3498db",
    "outlet": "#e74c3c",
}


class EdgePanel(QWidget):
    edit_requested = Signal(int)
    delete_requested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        layout.addWidget(QLabel("Edges"))
        self._list = QListWidget()
        layout.addWidget(self._list)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        edit_btn = QPushButton("Edit")
        edit_btn.clicked.connect(self._on_edit)
        btn_row.addWidget(edit_btn)
        del_btn = QPushButton("Reset")
        del_btn.clicked.connect(self._on_delete)
        btn_row.addWidget(del_btn)
        layout.addLayout(btn_row)

    def set_edges(self, edges: list[dict]):
        self._list.clear()
        for i, e in enumerate(edges):
            kind = e.get("kind", "wall")
            name = e.get("name", f"edge_{i}")
            color = _KIND_COLORS.get(kind, "#888888")
            item = QListWidgetItem(f"[{kind}] {name}")
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
