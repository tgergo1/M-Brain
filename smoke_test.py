import numpy as np
import json
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
from collections import Counter, defaultdict
import multiprocessing as mp
import time
import datetime
import types # Import the types module to check for modules

# --- Import the test configuration ---
from src import config_test as config
from src.dataset_generator import generate_dataset
from src.grid_cells import GridCellModule
from src.object_model import ObjectModel
from src.cortical_column import CorticalColumn
from src.cortex import Cortex

def log(message):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {message}")

def create_rotation_matrix(angles):
    return R.from_euler('xyz', angles, degrees=False).as_matrix()

def generate_sensory_sequence(obj_features, num_steps):
    feature_locations = list(obj_features.keys())
    if not feature_locations:
        return [], []
    feature_array = np.array(feature_locations)
    kdtree = cKDTree(feature_array)
    movements = np.random.normal(scale=0.2, size=(num_steps, 3))
    features = []
    current_pos = np.array(feature_locations[0])
    for move in movements:
        current_pos += move
        _, index = kdtree.query(current_pos)
        closest_feature_loc = feature_locations[index]
        features.append(obj_features[closest_feature_loc])
    return movements, features

def build_cortex_from_config(object_model) -> Cortex:
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

def training_worker(args):
    position, obj_name, instances = args
    object_model = ObjectModel()
    cortex = build_cortex_from_config(object_model)
    for instance_features in tqdm(instances, desc=f"  ↳ Core {position}: Train {obj_name}", position=position, leave=False):
        movements, features = generate_sensory_sequence(
            instance_features, config.SENSORY_STEPS_PER_OBJECT
        )
        cortex.process_sensory_sequence(movements, features, learn=True, obj_name=obj_name)
    return obj_name, cortex.columns[0].object_model.storage

def testing_worker(args):
    position, true_obj_name, instances, object_model_dict = args
    object_model = ObjectModel()
    object_model.storage = object_model_dict
    cortex = build_cortex_from_config(object_model)
    local_confusion_matrix = Counter()
    correct_count, total_count = 0, len(instances)
    for instance_features in tqdm(instances, desc=f"  ↳ Core {position}: Test {true_obj_name}", position=position, leave=False):
        movements, features = generate_sensory_sequence(
            instance_features, config.SENSORY_STEPS_PER_OBJECT
        )
        votes = cortex.process_sensory_sequence(movements, features, learn=False)
        if votes:
            predicted_obj_name = votes.most_common(1)[0][0]
            local_confusion_matrix.update([predicted_obj_name])
            if predicted_obj_name == true_obj_name:
                correct_count += 1
    return true_obj_name, local_confusion_matrix, correct_count, total_count

def main():
    start_time = time.time()
    log("--- SMOKE TEST: Quick Pipeline Verification (Stable Version) ---")

    log("Phase 1: Generating minimal datasets...")
    train_data = generate_dataset(config.NUM_OBJECT_TYPES, config.FEATURES_PER_OBJECT, config.DATASET_SIZE_TRAIN, desc="Test Train")
    test_data = generate_dataset(config.NUM_OBJECT_TYPES, config.FEATURES_PER_OBJECT, config.DATASET_SIZE_TEST, desc="Test Test")
    log("Phase 1: Dataset generation complete.")

    log("Phase 2: Parallel Training Start...")
    training_tasks = [(i + 1, name, inst) for i, (name, inst) in enumerate(train_data.items())]
    num_processes = min(mp.cpu_count(), len(training_tasks))
    log(f"Spawning {num_processes} workers for training...")
    with mp.Pool(processes=num_processes) as pool:
        training_results = list(tqdm(pool.imap_unordered(training_worker, training_tasks), total=len(training_tasks), desc="Overall Training"))
    log("Phase 2: Training Finished.")

    log("Phase 3: Aggregating models...")
    main_object_model = ObjectModel()
    for obj_name, storage_update in tqdm(training_results, desc="  ↳ Merging"):
        main_object_model.storage[obj_name] = storage_update.get(obj_name, [])
    log("Phase 3: Aggregation complete.")

    log("Phase 4: Parallel Testing Start...")
    testing_tasks = [(i + 1, name, inst, main_object_model.storage) for i, (name, inst) in enumerate(test_data.items())]
    log(f"Spawning {num_processes} workers for testing...")
    with mp.Pool(processes=num_processes) as pool:
        testing_results = list(tqdm(pool.imap_unordered(testing_worker, testing_tasks), total=len(testing_tasks), desc="Overall Testing"))
    log("Phase 4: Testing Finished.")

    log("Phase 5: Aggregating final results...")
    total_correct, total_preds = 0, 0
    final_cm = {name: Counter() for name in test_data.keys()}
    for true_obj_name, local_cm, correct, total in tqdm(testing_results, desc="  ↳ Consolidating"):
        total_correct += correct
        total_preds += total
        final_cm[true_obj_name].update(local_cm)
    log("Phase 5: Aggregation complete.")

    accuracy = (total_correct / total_preds) * 100 if total_preds > 0 else 0
    end_time = time.time()
    print("\n" + "="*40)
    log("SMOKE TEST COMPLETE")
    print("="*40)
    log(f"Total Execution Time: {end_time - start_time:.2f} seconds")
    log(f"Final Recognition Accuracy: {accuracy:.2f}% (Note: low accuracy is expected with minimal data)")
    print("="*40)

    # --- FIX: Intelligent serialization of the config file ---
    serializable_config = {}
    for key, value in vars(config).items():
        # Only include variables that are not modules and do not start with '__'
        if not key.startswith('__') and not isinstance(value, types.ModuleType):
            if isinstance(value, np.ndarray):
                serializable_config[key] = value.tolist()
            else:
                serializable_config[key] = value

    metrics = {
        "accuracy": accuracy,
        "correct_predictions": total_correct,
        "total_predictions": total_preds,
        "config": serializable_config,
        "confusion_matrix": {k: dict(v) for k, v in final_cm.items()}
    }
    log(f"Saving test metrics to {config.RESULTS_FILE}...")
    with open(config.RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)
    log("Save complete.")

if __name__ == "__main__":
    mp.freeze_support()
    main()