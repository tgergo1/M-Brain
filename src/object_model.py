from collections import defaultdict
from typing import Dict, List, Tuple
import random

class ObjectModel:
    """Stores feature-location mappings and now supports unlearning."""

    def __init__(self):
        self.storage: Dict[str, List[Tuple[Tuple[float, float, float], str]]] = defaultdict(list)

    def learn(self, obj: str, location: Tuple[float, float, float], feature: str) -> None:
        """Adds a feature-location association."""
        association = (tuple(location), feature)
        if association not in self.storage[obj]:
            self.storage[obj].append(association)

    def unlearn(self, obj: str, location: Tuple[float, float, float], feature: str) -> None:
        """Removes a feature-location association to weaken a memory."""
        association = (tuple(location), feature)
        if obj in self.storage and association in self.storage[obj]:
            self.storage[obj].remove(association)
            # If the object has no more associations, remove it entirely
            if not self.storage[obj]:
                del self.storage[obj]
    
    def query(self, location: Tuple[float, float, float], feature: str) -> List[str]:
        """Finds objects that have a given feature at a specific location."""
        results = []
        for obj, items in self.storage.items():
            for loc, feat in items:
                # Using a larger tolerance for location matching in noisy, real-world data
                if feat == feature and all(abs(a - b) < 0.5 for a, b in zip(loc, location)):
                    results.append(obj)
                    break
        return results

    def objects(self) -> List[str]:
        return list(self.storage.keys())