from typing import List, Dict
from collections import Counter
import random
from .cortical_column import CorticalColumn

class Cortex:
    """
    Collection of cortical columns that performs an iterative voting process to reach consensus.
    """
    def __init__(self, columns: List[CorticalColumn]):
        self.columns = columns
        # A simple proximity graph for lateral communication.
        self.column_neighbors = {i: random.sample(range(len(columns)), k=min(10, len(columns)-1))
                                 for i in range(len(columns))}

    def process_sensory_sequence(self, movement_sequence, feature_sequence, learn=False, obj_name=None):
        """
        Processes a sequence, either for learning (distributed) or for prediction (with consensus).
        """
        if learn:
            for movement, feature in zip(movement_sequence, feature_sequence):
                for col in self.columns:
                    col.process_input(movement, feature, learn=True, obj_name=obj_name)
            return None
        else:
            final_votes = Counter()
            for movement, feature in zip(movement_sequence, feature_sequence):
                for col in self.columns:
                    col.process_input(movement, feature, learn=False)

                consensus = self.reach_consensus()
                if consensus:
                    final_votes.update(consensus)
            return final_votes

    def reach_consensus(self, iterations: int = 3, initial_confidence: float = 0.5) -> Counter:
        """
        Simulates lateral communication between columns to converge on an answer.
        """
        column_hypotheses = [col.vote() for col in self.columns]
        confidence_scores = []
        for hypotheses in column_hypotheses:
            scores = {obj: initial_confidence for obj in hypotheses}
            confidence_scores.append(scores)

        for _ in range(iterations):
            new_confidence_scores = [scores.copy() for scores in confidence_scores]
            for i, col_scores in enumerate(confidence_scores):
                for obj in col_scores:
                    neighbor_support = 0
                    for neighbor_idx in self.column_neighbors[i]:
                        if obj in confidence_scores[neighbor_idx]:
                            neighbor_support += confidence_scores[neighbor_idx][obj]

                    total_possible_support = sum(initial_confidence for _ in self.column_neighbors[i])
                    if total_possible_support > 0:
                        reinforcement = neighbor_support / total_possible_support
                        new_confidence_scores[i][obj] *= (1 + reinforcement) / 2

            confidence_scores = new_confidence_scores

        final_votes = Counter()
        for scores in confidence_scores:
            if scores:
                best_obj = max(scores, key=scores.get)
                final_votes.update([best_obj])

        return final_votes