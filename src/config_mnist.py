# --- Dataset Parameters ---
DATASET_SIZE_TRAIN = 30
DATASET_SIZE_TEST = 10

# --- Cortex Parameters ---
NUM_CORTICAL_COLUMNS = 100
GRID_CELL_MODULES = [
    {'scale': 5.0}, {'scale': 8.0}, {'scale': 13.0}, {'scale': 21.0}
]

# --- Sensory Parameters ---
SENSORY_STEPS_PER_OBJECT = 100
PATCH_SIZE = 5

# --- Targeted Feedback Loop Parameters ---
REINFORCEMENT_RATE = 0.15
UNLEARNING_RATE = 0.30

# --- LOGGING & MODEL PERSISTENCE ---
MODEL_FILE = "models/mnist_final_model.pkl"
# Directory to store accuracy logs from each worker
LOG_DIR = "training_logs/"
# How often (in number of images) each worker should log its accuracy
LOG_FREQUENCY = 50