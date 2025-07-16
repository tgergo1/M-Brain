from typing import List, Tuple, Dict
from collections import Counter, defaultdict
import random
from .cortical_column import CorticalColumn

class Cortex:
    def __init__(self, columns: List[CorticalColumn]):
        self.columns = columns

    def process_sensory_sequence(self, movement_sequence, feature_sequence, learn=False, obj_name=None, return_active_columns=False):
        if learn:
            for movement, feature in zip(movement_sequence, feature_sequence):
                for col in self.columns:
                    col.process_input(movement, feature, learn=True, obj_name=obj_name)
            return
        
        # Prediction mode
        final_votes = Counter()
        active_columns_by_vote = defaultdict(list)

        for movement, feature in zip(movement_sequence, feature_sequence):
            for i, col in enumerate(self.columns):
                col.process_input(movement, feature, learn=False)
                # Get votes from this column for the current feature
                votes = col.vote()
                for vote in votes:
                    final_votes.update([vote])
                    # Track that this column (i) voted for this object (vote)
                    if i not in active_columns_by_vote[vote]:
                        active_columns_by_vote[vote].append(i)
        
        if return_active_columns:
            return final_votes, active_columns_by_vote
        else:
            return final_votes

    def apply_targeted_feedback(self, locations: List[Tuple], features: List[str], obj_name: str, column_indices: List[int], learn_rate: float):
        """Applies feedback ONLY to the specified columns."""
        if not column_indices:
            return

        # Apply feedback to a random subset of the feature-location pairs
        num_feedback_steps = int(abs(learn_rate) * len(features))
        for _ in range(num_feedback_steps):
            idx = random.randint(0, len(features) - 1)
            loc, feat = locations[idx], features[idx]
            
            # Apply feedback to a random column from the active set
            col_idx = random.choice(column_indices)
            column_to_modify = self.columns[col_idx]
            
            if learn_rate < 0: # Unlearning
                column_to_modify.layer2_3.object_model.unlearn(obj_name, loc, feat)
            else: # Reinforcement
                column_to_modify.layer2_3.object_model.learn(obj_name, loc, feat)