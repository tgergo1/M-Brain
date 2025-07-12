import numpy as np
import json
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
from collections import Counter
import multiprocessing as mp
import time
import datetime

from src import config
from src.dataset_generator import generate_dataset
from src.grid_cells import GridCellModule
from src.object_model import ObjectModel
from src.cortical_column import CorticalColumn
from src.cortex import Cortex

# --- Logging Function ---
def log(message):
    """Prints a message with a timestamp."""
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {message}")

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

def build_cortex_from_config(object_model) -> Cortex:
    """Builds a cortex using a shared object model."""
    columns = []
    for _ in range(config.NUM_CORTICAL_COLUMNS):
        modules = [
            GridCellModule(
                scale=m['scale'],
                orientation=create_rotation_matrix(m['orientation_angles'])
            )
            for m in config.GRID_CELL_MODULES
        ]
        columns.append(CorticalColumn(object_model, modules))
    return Cortex(columns)

# --- Worker Function for Training ---
def training_worker(args):
    """Worker for the training phase. Learns feature-location pairs."""
    position, obj_name, instances, object_model_dict = args
    
    # Reconstruct the object model in the worker
    object_model = ObjectModel()
    object_model.storage = object_model_dict
    
    cortex = build_cortex_from_config(object_model)

    for instance_features in tqdm(instances, desc=f"  ↳ Core {position}: Train {obj_name}", position=position, leave=False):
        movements, features = generate_sensory_sequence(
            instance_features, config.SENSORY_STEPS_PER_OBJECT, config.MOVEMENT_STD_DEV
        )
        cortex.process_sensory_sequence(movements, features, learn=True, obj_name=obj_name)
    
    # Return the updated storage
    return cortex.object_model.storage

# --- Worker Function for Testing ---
def testing_worker(args):
    """Worker for the testing phase. Predicts objects."""
    position, true_obj_name, instances, object_model_dict = args
    
    object_model = ObjectModel()
    object_model.storage = object_model_dict
    cortex = build_cortex_from_config(object_model)
    
    local_confusion_matrix = Counter()
    correct = 0
    total = 0

    for instance_features in tqdm(instances, desc=f"  ↳ Core {position}: Test {true_obj_name}", position=position, leave=False):
        movements, features = generate_sensory_sequence(
            instance_features, config.SENSORY_STEPS_PER_OBJECT, config.MOVEMENT_STD_DEV
        )
        votes = cortex.process_sensory_sequence(movements, features, learn=False)
        
        if votes:
            predicted_obj_name = votes.most_common(1)[0][0]
            local_confusion_matrix.update([predicted_obj_name])
            if predicted_obj_name == true_obj_name:
                correct += 1
        total += 1

    return true_obj_name, local_confusion_matrix, correct, total


def main():
    start_time = time.time()
    log("--- Simulation Start ---")

    # 1. Generate Datasets
    log("Phase 1: Generating datasets...")
    train_data = generate_dataset(config.NUM_OBJECT_TYPES, config.FEATURES_PER_OBJECT, config.DATASET_SIZE_TRAIN, desc="Train")
    test_data = generate_dataset(config.NUM_OBJECT_TYPES, config.FEATURES_PER_OBJECT, config.DATASET_SIZE_TEST, desc="Test")
    log("Phase 1: Dataset generation complete.")

    # 2. Training Phase
    log("Phase 2: Parallel Training Start...")
    object_model = ObjectModel()
    training_tasks = [(i + 1, name, inst, object_model.storage) for i, (name, inst) in enumerate(train_data.items())]
    
    num_processes = min(mp.cpu_count(), len(training_tasks))
    log(f"Spawning {num_processes} processes for training...")
    
    with mp.Pool(processes=num_processes) as pool:
        # The main progress bar tracks the completion of tasks by the workers
        training_results = list(tqdm(pool.imap_unordered(training_worker, training_tasks), total=len(training_tasks), desc="Overall Training Progress"))

    log("Phase 2: Parallel Training Finished. All cores have completed their tasks.")

    # Aggregate training results into a single object model
    log("Phase 3: Aggregating Learned Models...")
    for storage_update in tqdm(training_results, desc="  ↳ Merging learned data"):
        for obj, features in storage_update.items():
            object_model.storage[obj].extend(features)
    log("Phase 3: Model aggregation complete.")


    # 4. Testing Phase
    log("Phase 4: Parallel Testing Start...")
    testing_tasks = [(i + 1, name, inst, object_model.storage) for i, (name, inst) in enumerate(test_data.items())]
    
    with mp.Pool(processes=num_processes) as pool:
        testing_results = list(tqdm(pool.imap_unordered(testing_worker, testing_tasks), total=len(testing_tasks), desc="Overall Testing Progress "))

    log("Phase 4: Parallel Testing Finished. All cores have completed their tasks.")

    # 5. Final Aggregation of Test Results
    log("Phase 5: Aggregating Test Results...")
    total_correct_predictions = 0
    total_predictions = 0
    final_confusion_matrix = {name: Counter() for name in test_data.keys()}

    for true_obj_name, local_cm, correct, total in tqdm(testing_results, desc="  ↳ Final result consolidation"):
        total_correct_predictions += correct
        total_predictions += total
        final_confusion_matrix[true_obj_name].update(local_cm)
    log("Phase 5: Test result aggregation complete.")


    # 6. Final Report
    accuracy = (total_correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    end_time = time.time()
    
    print("\n" + "="*40)
    log("SIMULATION COMPLETE")
    print("="*40)
    log(f"Total Execution Time: {end_time - start_time:.2f} seconds")
    log(f"Final Recognition Accuracy: {accuracy:.2f}%")
    print("="*40)

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

    log(f"Saving final metrics to {config.RESULTS_FILE}...")
    with open(config.RESULTS_FILE, 'w') as f:
        json.dump(metrics, f, indent=4)
    log("Save complete.")
    log("To visualize results, run: python plot_results.py")


if __name__ == "__main__":
    mp.freeze_support()
    main()