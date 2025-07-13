import numpy as np

# --- SMOKE TEST PARAMETERS ---
NUM_OBJECT_TYPES = 2
FEATURES_PER_OBJECT = 10
DATASET_SIZE_TRAIN = 4
DATASET_SIZE_TEST = 2

# --- Cortex Parameters ---
NUM_CORTICAL_COLUMNS = 10
GRID_CELL_MODULES = [
    {'scale': 1.5, 'orientation_angles': (0, 0, 0)},
    {'scale': 2.0, 'orientation_angles': (0, np.pi/4, 0)},
]

# --- Training Parameters ---
SENSORY_STEPS_PER_OBJECT = 5

# --- Output ---
RESULTS_FILE = "results/smoketest_metrics.json"