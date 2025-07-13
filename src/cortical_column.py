from typing import List, Tuple, Dict
import jax.numpy as jnp
import numpy as np

from .grid_cells import GridCellSystem
from .object_model import ObjectModel

class Layer4:
    """Represents Layer 4, the primary sensory input layer."""
    def receive_sensory_input(self, feature: str) -> str:
        return feature

class Layer6:
    """Represents Layer 6, which handles location and motor context."""
    def __init__(self, grid_cell_system: GridCellSystem):
        self.location_system = grid_cell_system
        self._current_location = jnp.zeros(3)

    def update_location(self, movement: jnp.ndarray) -> jnp.ndarray:
        """Updates location based on movement (efference copy)."""
        self.location_system.update(movement)
        loc_3d_phase = self.location_system.get_state()[:3]
        scale = self.location_system.modules[0].scale if self.location_system.modules else 1.0
        self._current_location = loc_3d_phase * scale
        return self._current_location

    def get_current_location(self) -> jnp.ndarray:
        return self._current_location

class Layer2_3:
    """Represents Layers 2/3, where feature-location binding and prediction occurs."""
    def __init__(self, object_model: ObjectModel):
        self.object_model = object_model

    def learn(self, location: jnp.ndarray, feature: str, obj_name: str):
        # Convert JAX array to tuple for dictionary key
        location_tuple = tuple(np.round(np.asarray(location), 2))
        self.object_model.learn(obj_name, location_tuple, feature)

    def predict(self, location: jnp.ndarray, feature: str) -> List[str]:
        """Generate predictions (votes) based on the current sensory input and location."""
        if feature is None:
            return []
        location_tuple = tuple(np.round(np.asarray(location), 2))
        return self.object_model.query(location_tuple, feature)

class CorticalColumn:
    """A layered Cortical Column using JAX for its location system."""
    def __init__(self, modules: List):
        self_object_model = ObjectModel()
        grid_system = GridCellSystem(modules)

        self.layer4 = Layer4()
        self.layer6 = Layer6(grid_system)
        self.layer2_3 = Layer2_3(self_object_model)

        self.current_feature = None
        self.current_location = jnp.zeros(3)

    def process_input(self, movement: jnp.ndarray, feature: str, learn: bool = False, obj_name: str = None):
        """Processes a single timestep of sensory input and movement."""
        self.current_location = self.layer6.update_location(movement)
        self.current_feature = self.layer4.receive_sensory_input(feature)

        if learn and obj_name is not None and self.current_feature is not None:
            self.layer2_3.learn(self.current_location, self.current_feature, obj_name)

    def vote(self) -> List[str]:
        """Generates a vote for object identity based on current state."""
        return self.layer2_3.predict(self.current_location, self.current_feature)