from typing import List, Tuple

import numpy as np

from .grid_cells import GridCellSystem
from .object_model import ObjectModel


class CorticalColumn:
    """Simplified cortical column implementing feature-location modeling."""

    def __init__(self, object_model: ObjectModel, modules: List):
        self.object_model = object_model
        self.location_system = GridCellSystem(modules)
        self._current_location = (0, 0)

    def _get_location_from_state(self) -> Tuple[int, int]:
        """Derives a 2D integer location from the grid cell system's state."""
        # For simplicity, we use the phase of the first module (first 2 elements of state)
        # as the primary location signal.
        loc_2d = self.location_system.get_state()[:2]
        # We can use the scale of the first module to get a more "world-like" coordinate
        world_loc = loc_2d * self.location_system.modules[0].scale
        return tuple(np.round(world_loc).astype(int))

    def sense(self, movement: np.ndarray, feature: str, learn: bool = False, obj: str = None):
        # update location via path integration
        self.location_system.update(np.asarray(movement))
        self._current_location = self._get_location_from_state()

        if learn and obj is not None:
            self.object_model.learn(obj, self._current_location, feature)

    def predict_objects(self, feature: str) -> List[str]:
        # Prediction uses the column's current internal location
        return self.object_model.query(self._current_location, feature)