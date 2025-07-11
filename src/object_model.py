from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np


class ObjectModel:
    """Stores feature-location mappings for objects."""

    def __init__(self):
        # object -> list of (location tuple, feature)
        self.storage: Dict[str, List[Tuple[Tuple[int, int], str]]] = defaultdict(list)

    def learn(self, obj: str, location: Tuple[int, int], feature: str) -> None:
        self.storage[obj].append((tuple(location), feature))

    def query(self, location: Tuple[int, int], feature: str) -> List[str]:
        results = []
        for obj, items in self.storage.items():
            for loc, feat in items:
                if loc == tuple(location) and feat == feature:
                    results.append(obj)
                    break
        return results

    def objects(self) -> List[str]:
        return list(self.storage.keys())

