from typing import List

from .cortical_column import CorticalColumn


class Cortex:
    """Collection of cortical columns performing voting."""

    def __init__(self, columns: List[CorticalColumn]):
        self.columns = columns

    def vote(self, feature_inputs: List[str]) -> List[str]:
        assert len(feature_inputs) == len(self.columns)
        candidate_sets = []
        for col, feat in zip(self.columns, feature_inputs):
            candidate_sets.append(set(col.predict_objects(feat)))
        if not candidate_sets:
            return []
        consensus = set.intersection(*candidate_sets)
        if consensus:
            return sorted(consensus)
        else:
            # fallback: union of candidates
            union = set().union(*candidate_sets)
            return sorted(union)

