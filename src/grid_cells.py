import numpy as np

class GridCellModule:
    """A simple grid cell module implementing path integration."""

    def __init__(self, scale: float, orientation: float = 0.0):
        self.scale = float(scale)
        self.orientation = float(orientation)
        self.phase = np.zeros(2)
        # rotation matrix for orientation
        c, s = np.cos(orientation), np.sin(orientation)
        self.R = np.array([[c, -s], [s, c]])

    def update(self, movement: np.ndarray):
        movement = np.asarray(movement)
        rotated = self.R @ movement
        self.phase = (self.phase + rotated / self.scale) % 1.0

    def get_phase(self) -> np.ndarray:
        return self.phase.copy()


class GridCellSystem:
    """Collection of multiple grid cell modules."""

    def __init__(self, modules):
        self.modules = modules

    def update(self, movement: np.ndarray):
        for m in self.modules:
            m.update(movement)

    def get_state(self) -> np.ndarray:
        return np.concatenate([m.get_phase() for m in self.modules])

