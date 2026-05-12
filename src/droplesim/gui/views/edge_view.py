"""Tab 2: Edge clicking + labeling — edges only, no solid mask."""

from __future__ import annotations

import logging

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QWidget

from droplesim.gui.dialogs.edge_dialog import EdgeDialog
from droplesim.gui.panels.edge_panel import EdgePanel

log = logging.getLogger(__name__)

_EDGE_COLORS = {
    "wall": (150, 150, 150, 220),
    "inlet": (52, 152, 219, 255),   # #3498db
    "outlet": (231, 76, 60, 255),   # #e74c3c
}


class EdgeView(QWidget):
    edges_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._plot = pg.PlotWidget(title="Edge Labeling  (click edge to label)")
        self._plot.setBackground("#1a1a1a")
        self._plot.setAspectLocked(True)
        self._plot.setLabel("bottom", "x [µm]")
        self._plot.setLabel("left", "y [µm]")
        self._plot.scene().sigMouseClicked.connect(self._on_click)
        layout.addWidget(self._plot, stretch=3)

        self._panel = EdgePanel()
        self._panel.edit_requested.connect(self._on_edit_edge)
        self._panel.delete_requested.connect(self._on_delete_edge)
        layout.addWidget(self._panel, stretch=1)

        self._edge_curves: list[pg.PlotDataItem] = []
        self._arrow_items: list[pg.ArrowItem | None] = []
        self._edges: list[dict] = []
        self._edge_polylines_um: list[list[tuple[float, float]]] = []
        self._dx_um = 2.5
        self._origin_um = (0.0, 0.0)
        self._solid_mask = None
        self._channel_depth_um = 100.0

    def set_channel_depth(self, depth_um: float):
        self._channel_depth_um = depth_um

    def set_geometry(
        self,
        solid_mask: np.ndarray,
        dx_um: float,
        origin_um: tuple[float, float],
        edge_polylines_mm: list[list[tuple[float, float]]],
    ):
        self._solid_mask = solid_mask
        self._dx_um = dx_um
        self._origin_um = origin_um

        self._edge_polylines_um = []
        self._edges = []
        for i, poly_mm in enumerate(edge_polylines_mm):
            pts_um = [(x * 1000.0, y * 1000.0) for x, y in poly_mm]
            self._edge_polylines_um.append(pts_um)
            self._edges.append({
                "name": f"edge_{i}",
                "kind": "wall",
                "points_um": pts_um,
                "phi": 1.0,
                "ux": 0.0,
                "uy": 0.0,
                "flow_rate": 0.0,
            })

        self._redraw_edges()
        self._panel.set_edges(self._edges)
        self._plot.autoRange()
        log.info("Edge view: %d edges loaded", len(self._edges))

    def _redraw_edges(self):
        for c in self._edge_curves:
            self._plot.removeItem(c)
        self._edge_curves.clear()
        for a in self._arrow_items:
            if a is not None:
                self._plot.removeItem(a)
        self._arrow_items.clear()

        for i, edge in enumerate(self._edges):
            pts = edge["points_um"]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            color = _EDGE_COLORS.get(edge["kind"], (150, 150, 150, 220))
            width = 3 if edge["kind"] != "wall" else 1.5
            curve = self._plot.plot(xs, ys, pen=pg.mkPen(color=color, width=width))
            self._edge_curves.append(curve)

            # Draw flow direction arrow for inlets with flow_rate > 0
            if edge["kind"] == "inlet" and edge.get("flow_rate", 0) > 0:
                ndir = self._edge_normal_dir(i)
                p0 = np.array(pts[0])
                p1 = np.array(pts[-1])
                mid = (p0 + p1) / 2
                arrow_len = 10 * self._dx_um
                tip = mid + np.array(ndir) * arrow_len
                angle = np.degrees(np.arctan2(-ndir[1], -ndir[0]))
                arrow = pg.ArrowItem(
                    pos=(tip[0], tip[1]),
                    angle=angle,
                    headLen=8,
                    headWidth=6,
                    tailLen=arrow_len * 0.6,
                    tailWidth=2,
                    pen=pg.mkPen(color=(52, 152, 219, 200), width=1),
                    brush=(52, 152, 219, 180),
                )
                self._plot.addItem(arrow)
                self._arrow_items.append(arrow)
            else:
                self._arrow_items.append(None)

    def _on_click(self, event):
        if not self._edges:
            return
        pos = event.scenePos()
        mouse_pt = self._plot.plotItem.vb.mapSceneToView(pos)
        mx, my = mouse_pt.x(), mouse_pt.y()

        best_idx = -1
        best_dist = float("inf")
        threshold = 5 * self._dx_um

        for i, edge in enumerate(self._edges):
            pts = edge["points_um"]
            for k in range(len(pts) - 1):
                d = self._point_to_segment_dist(
                    mx, my, pts[k][0], pts[k][1], pts[k + 1][0], pts[k + 1][1]
                )
                if d < best_dist:
                    best_dist = d
                    best_idx = i

        if best_idx >= 0 and best_dist < threshold:
            self._show_edge_dialog(best_idx)

    @staticmethod
    def _point_to_segment_dist(px, py, x1, y1, x2, y2) -> float:
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
        cx, cy = x1 + t * dx, y1 + t * dy
        return ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5

    def _edge_width_um(self, idx: int) -> float:
        pts = self._edges[idx]["points_um"]
        if len(pts) < 2:
            return 0.0
        p0 = np.array(pts[0])
        p1 = np.array(pts[-1])
        return float(np.linalg.norm(p1 - p0))

    def _edge_normal_dir(self, idx: int) -> tuple[float, float]:
        pts = self._edges[idx]["points_um"]
        if len(pts) < 2:
            return (1.0, 0.0)
        p0 = np.array(pts[0])
        p1 = np.array(pts[-1])
        tangent = p1 - p0
        length = np.linalg.norm(tangent)
        if length < 1e-12:
            return (1.0, 0.0)
        tangent /= length
        nx, ny = -tangent[1], tangent[0]
        if self._solid_mask is not None:
            mid = (p0 + p1) / 2
            ox, oy = self._origin_um
            test_pt = mid + np.array([nx, ny]) * self._dx_um * 3
            ix = int((test_pt[0] - ox) / self._dx_um)
            iy = int((test_pt[1] - oy) / self._dx_um)
            h, w = self._solid_mask.shape
            # OOB counts as solid — edges at grid boundary always have
            # their test point land outside, so we must flip in that case.
            is_fluid = (0 <= iy < h and 0 <= ix < w and not self._solid_mask[iy, ix])
            if not is_fluid:
                nx, ny = -nx, -ny
        # User override: flip if auto-detection got it wrong
        if self._edges[idx].get("normal_flipped", False):
            nx, ny = -nx, -ny
        return (nx, ny)

    def _show_edge_dialog(self, idx: int):
        edge = self._edges[idx]
        width_um = self._edge_width_um(idx)
        dlg = EdgeDialog(
            self,
            name=edge.get("name", ""),
            kind=edge.get("kind", "wall"),
            phi=edge.get("phi", 1.0),
            flow_rate=edge.get("flow_rate", 0.0),
            edge_width_um=width_um,
            channel_depth_um=self._channel_depth_um,
            contact_angle_deg=edge.get("contact_angle_deg"),
            outlet_bc=edge.get("outlet_bc", "pressure"),
            rho_target=edge.get("rho_target", 1.0),
            normal_flipped=edge.get("normal_flipped", False),
        )
        if dlg.exec():
            data = dlg.result_data()
            edge["name"] = data["name"]
            edge["kind"] = data["kind"]
            edge["phi"] = data.get("phi", 1.0)
            edge["flow_rate"] = data.get("flow_rate", 0.0)
            edge["contact_angle_deg"] = data.get("contact_angle_deg")
            edge["outlet_bc"] = data.get("outlet_bc", "pressure")
            edge["rho_target"] = data.get("rho_target", 1.0)
            edge["normal_flipped"] = data.get("normal_flipped", False)

            if data.get("flow_rate", 0) > 0 and data["kind"] == "inlet":
                Q_m3s = data["flow_rate"] * 1e-9 / 60.0
                width_m = width_um * 1e-6
                depth_m = self._channel_depth_um * 1e-6
                area = width_m * depth_m
                if area > 0:
                    u_ms = Q_m3s / area
                    ndir = self._edge_normal_dir(idx)
                    edge["ux"] = u_ms * ndir[0]
                    edge["uy"] = u_ms * ndir[1]
                    flipped = edge.get("normal_flipped", False)
                    log.info(
                        "Edge '%s': Q=%.2f µL/min -> u=%.4e m/s, normal=(%.3f,%.3f), "
                        "ux=%.4e uy=%.4e, flipped=%s",
                        data["name"], data["flow_rate"], u_ms,
                        ndir[0], ndir[1], edge["ux"], edge["uy"], flipped,
                    )
            else:
                edge["ux"] = 0.0
                edge["uy"] = 0.0

            self._redraw_edges()
            self._panel.set_edges(self._edges)
            self.edges_changed.emit()

    def _on_edit_edge(self, idx: int):
        if 0 <= idx < len(self._edges):
            self._show_edge_dialog(idx)

    def _on_delete_edge(self, idx: int):
        if 0 <= idx < len(self._edges):
            self._edges[idx]["kind"] = "wall"
            self._edges[idx]["name"] = f"edge_{idx}"
            self._edges[idx]["flow_rate"] = 0.0
            self._edges[idx]["ux"] = 0.0
            self._edges[idx]["uy"] = 0.0
            self._redraw_edges()
            self._panel.set_edges(self._edges)
            self.edges_changed.emit()

    def get_edges(self) -> list[dict]:
        return self._edges

    def set_edges_from_state(self, edges: list[dict]):
        if len(edges) == len(self._edges):
            for i, e in enumerate(edges):
                self._edges[i].update(e)
                self._edges[i]["points_um"] = self._edge_polylines_um[i]
                # Recalculate ux/uy from flow_rate + corrected normal
                if e.get("kind") == "inlet" and e.get("flow_rate", 0) > 0:
                    width_um = self._edge_width_um(i)
                    Q_m3s = e["flow_rate"] * 1e-9 / 60.0
                    width_m = width_um * 1e-6
                    depth_m = self._channel_depth_um * 1e-6
                    area = width_m * depth_m
                    if area > 0:
                        u_ms = Q_m3s / area
                        ndir = self._edge_normal_dir(i)
                        self._edges[i]["ux"] = u_ms * ndir[0]
                        self._edges[i]["uy"] = u_ms * ndir[1]
                        flipped = self._edges[i].get("normal_flipped", False)
                        log.info(
                            "Edge '%s' (load): normal=(%.3f,%.3f), ux=%.4e uy=%.4e, flipped=%s",
                            e.get("name", f"edge_{i}"),
                            ndir[0], ndir[1],
                            self._edges[i]["ux"], self._edges[i]["uy"], flipped,
                        )
        self._redraw_edges()
        self._panel.set_edges(self._edges)
