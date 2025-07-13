import numpy as np

# --- Simulation Parameters ---
NUM_OBJECT_TYPES = 10
FEATURES_PER_OBJECT = 50
DATASET_SIZE_TRAIN = 100
DATASET_SIZE_TEST = 50

# --- Cortex Parameters ---
NUM_CORTICAL_COLUMNS = 1000
GRID_CELL_MODULES = [
    {'scale': 1.5, 'orientation_angles': (0, 0, 0)},
    {'scale': 2.0, 'orientation_angles': (0, np.pi/4, 0)},
    {'scale': 2.5, 'orientation_angles': (0, 0, np.pi/2)},
    {'scale': 3.0, 'orientation_angles': (np.pi/4, 0, np.pi/4)},
]

# --- Training Parameters ---
SENSORY_STEPS_PER_OBJECT = 100
MOVEMENT_STD_DEV = 0.2

# --- Output ---
RESULTS_FILE = "results/simulation_metrics.json"