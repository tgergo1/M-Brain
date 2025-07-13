import jax.numpy as jnp
from jax import jit

class GridCellModule:
    """A JAX-accelerated grid cell module implementing 3D path integration."""

    def __init__(self, scale: float, orientation: jnp.ndarray = jnp.identity(3)):
        self.scale = float(scale)
        self.orientation_matrix = jnp.asarray(orientation)
        self.phase = jnp.zeros(3)
        self._update_func = jit(self._static_update)

    @staticmethod
    def _static_update(phase: jnp.ndarray, movement_3d: jnp.ndarray, orientation_matrix: jnp.ndarray, scale: float):
        """A static version of the update logic for JAX JIT compilation."""
        rotated = orientation_matrix @ movement_3d
        new_phase = (phase + rotated / scale) % 1.0
        return new_phase

    def update(self, movement: jnp.ndarray):
        """Applies the JIT-compiled update function."""
        movement_3d = jnp.asarray(movement)
        if movement_3d.shape != (3,):
            raise ValueError("Movement vector must be 3D.")
        self.phase = self._update_func(self.phase, movement_3d, self.orientation_matrix, self.scale)

    def get_phase(self) -> jnp.ndarray:
        return self.phase

class GridCellSystem:
    """Collection of multiple grid cell modules."""
    def __init__(self, modules):
        self.modules = modules

    def update(self, movement: jnp.ndarray):
        """Updates all modules in the system."""
        for m in self.modules:
            m.update(movement)

    def get_state(self) -> jnp.ndarray:
        """Concatenates the state of all modules into a single JAX array."""
        return jnp.concatenate([m.get_phase() for m in self.modules])