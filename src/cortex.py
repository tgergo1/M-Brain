from typing import List, Dict
from collections import Counter
from .cortical_column import CorticalColumn

class Cortex:
    """Collection of cortical columns performing voting."""
    def __init__(self, columns: List[CorticalColumn]):
        self.columns = columns

    def process_sensory_sequence(self, movement_sequence, feature_sequence, learn=False, obj_name=None):
        """Processes a sequence of movements and features."""
        final_votes = Counter()
        for movement, feature in zip(movement_sequence, feature_sequence):
            predictions = []
            for col in self.columns:
                col.sense(movement, feature, learn=learn, obj=obj_name)
                if not learn:
                    predictions.extend(col.predict_objects(feature))
            if not learn:
                final_votes.update(predictions)
        return final_votes