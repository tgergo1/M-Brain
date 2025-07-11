from typing import List, Dict
from collections import Counter

from .cortical_column import CorticalColumn

class Cortex:
    """Collection of cortical columns performing voting."""

    def __init__(self, columns: List[CorticalColumn]):
        self.columns = columns

    def sense_and_learn(self, movement, feature_input: str, object_name: str):
        """Update all columns and instruct them to learn the feature-location pair."""
        for col in self.columns:
            col.sense(movement, feature_input, learn=True, obj=object_name)

    def sense_and_vote(self, movement, feature_input: str) -> Dict[str, int]:
        """Update all columns and aggregate their votes."""
        all_predictions = []
        for col in self.columns:
            # First, update the column's internal location based on movement
            col.sense(movement, feature_input, learn=False)
            # Then, ask it to predict based on its new location
            predictions = col.predict_objects(feature_input)
            all_predictions.extend(predictions)
        
        # Return a count of each object prediction
        return Counter(all_predictions)