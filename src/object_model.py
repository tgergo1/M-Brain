from collections import defaultdict
from typing import Dict, List, Tuple

class ObjectModel:
    """Stores feature-location mappings for objects."""

    def __init__(self):
        # object -> list of (location tuple (3D), feature)
        self.storage: Dict[str, List[Tuple[Tuple[float, float, float], str]]] = defaultdict(list)

    def learn(self, obj: str, location: Tuple[float, float, float], feature: str) -> None:
        # Check if this exact feature at this location is already learned for this object
        if (tuple(location), feature) not in self.storage[obj]:
            self.storage[obj].append((tuple(location), feature))

    def query(self, location: Tuple[float, float, float], feature: str) -> List[str]:
        results = []
        for obj, items in self.storage.items():
            for loc, feat in items:
                # Use a small tolerance for floating point comparison
                if feat == feature and all(abs(a - b) < 1e-9 for a, b in zip(loc, location)):
                    results.append(obj)
                    break
        return results

    def objects(self) -> List[str]:
        return list(self.storage.keys())