from typing import List, Tuple
import numpy as np

from .grid_cells import GridCellSystem
from .object_model import ObjectModel

class CorticalColumn:
    """3D cortical column implementing feature-location modeling."""
    def __init__(self, object_model: ObjectModel, modules: List):
        self.object_model = object_model
        self.location_system = GridCellSystem(modules)
        self._current_location = (0.0, 0.0, 0.0)

    def _get_location_from_state(self) -> Tuple[float, float, float]:
        """Derives a 3D location from the high-dimensional grid cell state."""
        # For simplicity, we'll use the phase of the first module as the base location.
        # A more complex model might combine phases from multiple modules.
        loc_3d_phase = self.location_system.get_state()[:3]
        world_loc = loc_3d_phase * self.location_system.modules[0].scale
        return tuple(np.round(world_loc, 2))

    def sense(self, movement: np.ndarray, feature: str, learn: bool = False, obj: str = None):
        self.location_system.update(np.asarray(movement))
        self._current_location = self._get_location_from_state()

        if learn and obj is not None and feature is not None:
            self.object_model.learn(obj, self._current_location, feature)

    def predict_objects(self, feature: str) -> List[str]:
        if feature is None:
            return []
        return self.object_model.query(self._current_location, feature)