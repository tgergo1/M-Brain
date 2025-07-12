import numpy as np
import json
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
from collections import Counter
import multiprocessing as mp
import time

from src import config
from src.dataset_generator import generate_dataset
from src.grid_cells import GridCellModule
from src.object_model import ObjectModel
from src.cortical_column import CorticalColumn
from src.cortex import Cortex

# --- Helper Functions ---
def create_rotation_matrix(angles):
    return R.from_euler('xyz', angles, degrees=False).as_matrix()

def generate_sensory_sequence(obj_features, num_steps, move_std_dev):
    feature_locations = list(obj_features.keys())
    if not feature_locations:
        return [], []
    feature_array = np.array(feature_locations)
    kdtree = cKDTree(feature_array)
    movements = np.random.normal(scale=move_std_dev, size=(num_steps, 3))
    features = []
    current_pos = np.array(feature_locations[0])
    for move in movements:
        current_pos += move
        _, index = kdtree.query(current_pos)
        closest_feature_loc = feature_locations[index]
        features.append(obj_features[closest_feature_loc])
    return movements, features

def build_cortex_from_config() -> Cortex:
    """Helper to build a cortex instance based on config."""
    model = ObjectModel()
    columns = []
    for _ in range(config.NUM_CORTICAL_COLUMNS):
        modules = [
            GridCellModule(
                scale=m['scale'],
                orientation=create_rotation_matrix(m['orientation_angles'])
            )
            for m in config.GRID_CELL_MODULES
        ]
        columns.append(CorticalColumn(model, modules))
    return Cortex(columns)

# --- Worker Function for Parallel Processing (with Position for TQDM) ---
def worker_task(args):
    """
    This function is executed by each worker process. It now takes a position
    argument to manage its own detailed progress bar.
    """
    position, obj_name, instances, test_data = args
    
    # Each worker gets its own Cortex instance
    cortex = build_cortex_from_config()

    # --- Training Phase for this worker ---
    # This tqdm bar is specific to this worker and will be displayed on its assigned line
    for instance_features in tqdm(instances, desc=f"Core {position}: Train {obj_name}", position=position, leave=False):
        movements, features = generate_sensory_sequence(
            instance_features, config.SENSORY_STEPS_PER_OBJECT, config.MOVEMENT_STD_DEV
        )
        cortex.process_sensory_sequence(movements, features, learn=True, obj_name=obj_name)
    
    # --- Testing phase for this worker ---
    local_confusion_matrix = Counter()
    correct = 0
    total = 0

    # Test this worker's learned object against all test instances of the same class
    test_instances = test_data.get(obj_name, [])
    for instance_features in test_instances:
        movements, features = generate_sensory_sequence(
            instance_features, config.SENSORY_STEPS_PER_OBJECT, config.MOVEMENT_STD_DEV
        )
        # We only need votes on the single object model this cortex has learned
        votes = cortex.process_sensory_sequence(movements, features, learn=False)
        
        if votes:
            predicted_obj_name = votes.most_common(1)[0][0]
            local_confusion_matrix.update([predicted_obj_name])
            if predicted_obj_name == obj_name:
                correct += 1
        total += 1

    return obj_name, local_confusion_matrix, correct, total

def main():
    start_time = time.time()
    print("--- Grand Scale M-Brain Simulation (Parallelized with Detailed Progress) ---")

    # 1. Generate Datasets
    print("Generating datasets...")
    train_data = generate_dataset(config.NUM_OBJECT_TYPES, config.FEATURES_PER_OBJECT, config.DATASET_SIZE_TRAIN, desc="Train")
    test_data = generate_dataset(config.NUM_OBJECT_TYPES, config.FEATURES_PER_OBJECT, config.DATASET_SIZE_TEST, desc="Test")

    # 2. Prepare tasks for the process pool, including a position for each worker's progress bar
    tasks = [(i + 1, obj_name, instances, test_data) for i, (obj_name, instances) in enumerate(train_data.items())]

    # 3. Execute Training and Testing in Parallel
    num_processes = min(mp.cpu_count(), len(tasks))
    print(f"\nStarting parallel processing on {num_processes} cores...")
    
    # Main (outer) progress bar
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(worker_task, tasks), total=len(tasks), desc="Overall Progress"))

    # 4. Aggregate Results
    print("\n\nAggregating results from all workers...")
    total_correct_predictions = 0
    total_predictions = 0
    final_confusion_matrix = {name: Counter() for name in test_data.keys()}

    for obj_name, local_cm, correct, total in results:
        total_correct_predictions += correct
        total_predictions += total
        final_confusion_matrix[obj_name].update(local_cm)

    # 5. Log Final Metrics
    accuracy = (total_correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    end_time = time.time()
    
    print("\n--- Simulation Complete ---")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    print(f"Final Recognition Accuracy: {accuracy:.2f}%")

    serializable_config = {k: v for k, v in vars(config).items() if not k.startswith('__')}
    for k, v in serializable_config.items():
        if isinstance(v, np.ndarray):
            serializable_config[k] = v.tolist()

    metrics = {
        "accuracy": accuracy,
        "correct_predictions": total_correct_predictions,
        "total_predictions": total_predictions,
        "config": serializable_config,
        "confusion_matrix": {k: dict(v) for k, v in final_confusion_matrix.items()}
    }

    print(f"Saving results to {config.RESULTS_FILE}...")
    with open(config.RESULTS_FILE, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print("\nTo visualize results, run: python plot_results.py")

if __name__ == "__main__":
    # This is crucial for multiprocessing to work correctly on all platforms
    mp.freeze_support()
    main()