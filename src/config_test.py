# src/config_test.py
import numpy as np

# --- SMOKE TEST PARAMETERS ---
# These settings are intentionally small to ensure a very fast run.

NUM_OBJECT_TYPES = 2        # Use only 2 object classes
FEATURES_PER_OBJECT = 10    # Only 10 features per object
DATASET_SIZE_TRAIN = 4      # Only 4 training instances per class
DATASET_SIZE_TEST = 2       # Only 2 testing instances per class

# --- Cortex Parameters ---
NUM_CORTICAL_COLUMNS = 10   # A very small number of columns
GRID_CELL_MODULES = [
    {'scale': 1.5, 'orientation_angles': (0, 0, 0)},
    {'scale': 2.0, 'orientation_angles': (0, np.pi/4, 0)},
]

# --- Training Parameters ---
SENSORY_STEPS_PER_OBJECT = 5 # Very few sensory steps

# --- Output ---
# Save to a separate file to not overwrite full simulation results
RESULTS_FILE = "results/smoketest_metrics.json"