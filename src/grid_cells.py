import numpy as np

class GridCellModule:
    """A simple grid cell module implementing 3D path integration."""

    def __init__(self, scale: float, orientation: np.ndarray = np.identity(3)):
        self.scale = float(scale)
        self.orientation_matrix = np.asarray(orientation)
        self.phase = np.zeros(3)

    def update(self, movement: np.ndarray):
        movement_3d = np.asarray(movement)
        if movement_3d.shape != (3,):
            raise ValueError("Movement vector must be 3D.")
        
        rotated = self.orientation_matrix @ movement_3d
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