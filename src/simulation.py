"""Demonstration of the Thousand Brains style modeling."""
import numpy as np

from .grid_cells import GridCellModule
from .object_model import ObjectModel
from .cortical_column import CorticalColumn
from .cortex import Cortex


# define objects as mapping location -> feature
OBJECTS = {
    "Square": {
        (0, 0): "corner",
        (1, 0): "edge",
        (0, 1): "edge",
        (1, 1): "corner",
    },
    "Triangle": {
        (0, 0): "corner",
        (1, 0): "edge",
        (0, 1): "edge",
    },
}


def build_object_model() -> ObjectModel:
    model = ObjectModel()
    for obj, feats in OBJECTS.items():
        for loc, feat in feats.items():
            model.learn(obj, loc, feat)
    return model


def build_cortex(num_columns: int = 3) -> Cortex:
    model = build_object_model()
    columns = []
    for _ in range(num_columns):
        modules = [GridCellModule(scale=1.0)]
        columns.append(CorticalColumn(model, modules))
    return Cortex(columns)


def demo():
    cortex = build_cortex()
    # simulate each column sensing a feature at a known location
    features = ["corner", "edge", "edge"]
    hypotheses = cortex.vote(features)
    print("Hypotheses:", hypotheses)


if __name__ == "__main__":
    demo()

