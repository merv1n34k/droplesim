"""Ring buffer for simulation frame history with HDF5 export."""

from __future__ import annotations

import dataclasses
from collections import deque

import numpy as np


@dataclasses.dataclass(slots=True)
class FrameRecord:
    step: int
    phi: np.ndarray
    rho: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    elapsed: float
    mlups: float
    extra: dict | None


class FrameBuffer:
    def __init__(self, maxlen: int = 1000):
        self._frames: deque[FrameRecord] = deque(maxlen=maxlen)

    def append(self, rec: FrameRecord) -> None:
        self._frames.append(rec)

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, idx: int) -> FrameRecord:
        return self._frames[idx]

    def clear(self) -> None:
        self._frames.clear()

    @property
    def maxlen(self) -> int:
        return self._frames.maxlen

    def export_hdf5(self, path: str) -> None:
        import h5py

        with h5py.File(path, "w") as f:
            n = len(self._frames)
            if n == 0:
                return
            steps = np.array([r.step for r in self._frames], dtype=np.int64)
            f.create_dataset("step", data=steps)
            f.create_dataset("phi", data=np.stack([r.phi for r in self._frames]))
            f.create_dataset("rho", data=np.stack([r.rho for r in self._frames]))
            f.create_dataset("ux", data=np.stack([r.ux for r in self._frames]))
            f.create_dataset("uy", data=np.stack([r.uy for r in self._frames]))
            f.create_dataset(
                "elapsed", data=np.array([r.elapsed for r in self._frames])
            )
            f.create_dataset(
                "mlups", data=np.array([r.mlups for r in self._frames])
            )

            extra_keys: set[str] = set()
            for r in self._frames:
                if r.extra:
                    extra_keys.update(r.extra.keys())
            for key in sorted(extra_keys):
                arrays = []
                for r in self._frames:
                    if r.extra and key in r.extra:
                        arrays.append(r.extra[key])
                    else:
                        arrays.append(np.zeros_like(self._frames[0].phi))
                f.create_dataset(f"extra/{key}", data=np.stack(arrays))
