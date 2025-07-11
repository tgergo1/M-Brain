from typing import List, Tuple

import numpy as np

from .grid_cells import GridCellModule, GridCellSystem
from .object_model import ObjectModel


class CorticalColumn:
    """Simplified cortical column implementing feature-location modeling."""

    def __init__(self, object_model: ObjectModel, modules: List[GridCellModule]):
        self.object_model = object_model
        self.location_system = GridCellSystem(modules)

    def sense(self, movement: Tuple[int, int], feature: str, learn: bool = False, obj: str = None):
        # update location via path integration
        self.location_system.update(np.asarray(movement))
        loc = tuple(np.round(self.location_system.get_state()[:2]).astype(int))
        if learn and obj is not None:
            self.object_model.learn(obj, loc, feature)
        return loc

    def predict_objects(self, feature: str) -> List[str]:
        loc = tuple(np.round(self.location_system.get_state()[:2]).astype(int))
        return self.object_model.query(loc, feature)

