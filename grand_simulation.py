import jax.numpy as jnp
from jax import random
from jax import config as jax_config
import numpy as np
import json
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
from collections import Counter
import multiprocessing as mp
import time
import datetime
import types
import os

jax_config.update("jax_enable_x64", True)

from src import config
from src.dataset_generator import generate_dataset
from src.grid_cells import GridCellModule
from src.cortical_column import CorticalColumn
from src.cortex import Cortex

def log(message):
    """Enhanced logger with a timestamp."""
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def create_rotation_matrix(angles):
    return jnp.array(R.from_euler('xyz', angles, degrees=False).as_matrix())

def generate_sensory_sequence(obj_features, num_steps, move_std_dev, key):
    """Generates sensory sequences using JAX for random number generation."""
    feature_locations = list(obj_features.keys())
    if not feature_locations:
        return [], [], key
    feature_array_np = np.array(feature_locations)
    kdtree = cKDTree(feature_array_np)

    key, subkey = random.split(key)
    movements = random.normal(subkey, shape=(num_steps, 3)) * move_std_dev
    
    features = []
    current_pos = feature_array_np[0]
    for move in movements:
        current_pos += np.asarray(move)
        _, index = kdtree.query(current_pos)
        closest_feature_loc = feature_locations[index]
        features.append(obj_features[closest_feature_loc])
    return movements, features, key

def build_cortex_from_config() -> Cortex:
    """Builds a cortex with JAX-ready columns."""
    columns = []
    for _ in range(config.NUM_CORTICAL_COLUMNS):
        modules = [
            GridCellModule(
                scale=m['scale'],
                orientation=create_rotation_matrix(m['orientation_angles'])
            ) for m in config.GRID_CELL_MODULES
        ]
        columns.append(CorticalColumn(modules))
    return Cortex(columns)

def training_worker(args):
    """
    Worker trains a cortex and returns ONLY the learned data (the object model's storage),
    not the entire heavy cortex object.
    """
    position, obj_name, instances, seed = args
    pid = os.getpid()
    log(f"[Worker PID: {pid}, Core: {position}] Starting training for '{obj_name}'")
    
    key = random.PRNGKey(seed)
    cortex = build_cortex_from_config()

    for instance_features in tqdm(instances, desc=f"  ↳ PID {pid} Training '{obj_name}'", position=position, leave=False, ncols=100):
        key, subkey = random.split(key)
        movements, features, key = generate_sensory_sequence(
            instance_features, config.SENSORY_STEPS_PER_OBJECT, config.MOVEMENT_STD_DEV, subkey
        )
        cortex.process_sensory_sequence(movements, features, learn=True, obj_name=obj_name)

    # **CRITICAL CHANGE**: Extract and return ONLY the learned data from each column.
    learned_data = [col.layer2_3.object_model.storage for col in cortex.columns]
    log(f"[Worker PID: {pid}, Core: {position}] Finished training for '{obj_name}', returning learned data.")
    return learned_data

def testing_worker(args):
    """Worker tests against the master cortex using JAX."""
    position, true_obj_name, instances, master_cortex, seed = args
    pid = os.getpid()
    log(f"[Worker PID: {pid}, Core: {position}] Starting testing for '{true_obj_name}'")
    
    key = random.PRNGKey(seed)
    local_confusion_matrix = Counter()
    correct_count, total_count = 0, len(instances)

    for instance_features in tqdm(instances, desc=f"  ↳ PID {pid} Testing '{true_obj_name}'", position=position, leave=False, ncols=100):
        key, subkey = random.split(key)
        movements, features, key = generate_sensory_sequence(
            instance_features, config.SENSORY_STEPS_PER_OBJECT, config.MOVEMENT_STD_DEV, subkey
        )
        votes = master_cortex.process_sensory_sequence(movements, features, learn=False)
        if votes:
            predicted_obj_name = votes.most_common(1)[0][0]
            local_confusion_matrix.update([predicted_obj_name])
            if predicted_obj_name == true_obj_name:
                correct_count += 1
    return true_obj_name, local_confusion_matrix, correct_count, total_count

def main():
    """Main execution function, wrapped for safety."""
    mp.set_start_method('spawn', force=True)
    start_time = time.time()
    log("--- Grand Scale M-Brain Simulation (JAX/GPU Accelerated, Optimized IPC) ---")

    log("Phase 1: Generating datasets...")
    train_data = generate_dataset(config.NUM_OBJECT_TYPES, config.FEATURES_PER_OBJECT, config.DATASET_SIZE_TRAIN, desc="Train Dataset")
    test_data = generate_dataset(config.NUM_OBJECT_TYPES, config.FEATURES_PER_OBJECT, config.DATASET_SIZE_TEST, desc="Test Dataset")
    log("Phase 1 Complete.")

    log("Phase 2: Parallel Training of Expert Models...")
    num_processes = min(mp.cpu_count(), len(train_data))
    log(f"Spawning {num_processes} workers for JAX-based training...")
    training_tasks = [(i + 1, name, inst, int(start_time) + i) for i, (name, inst) in enumerate(train_data.items())]

    with mp.Pool(processes=num_processes) as pool:
        # This now receives a list of lists of dictionaries, which is much faster.
        all_learned_data = list(tqdm(pool.imap(training_worker, training_tasks), total=len(training_tasks), desc="Overall Training Progress", ncols=100))
    log("Phase 2 Complete.")

    log("Phase 3: Merging expert knowledge into Master Cortex...")
    master_cortex = build_cortex_from_config()
    # This loop is fast as it only merges the lightweight data structures.
    for expert_data_per_cortex in tqdm(all_learned_data, desc="Merging Knowledge", ncols=100):
        for i, column_data in enumerate(expert_data_per_cortex):
            # Update the master cortex's corresponding column with the learned data.
            master_cortex.columns[i].layer2_3.object_model.storage.update(column_data)
    log("Phase 3 Complete.")

    log("Phase 4: Parallel Testing against Master Cortex...")
    log("Distributing the Master Cortex to workers (this may take a moment)...")
    testing_tasks = [(i + 1, name, inst, master_cortex, int(start_time) + 1000 + i) for i, (name, inst) in enumerate(test_data.items())]
    num_processes = min(mp.cpu_count(), len(testing_tasks))
    log(f"Spawning {num_processes} workers for JAX-based testing...")
    
    with mp.Pool(processes=num_processes) as pool:
        testing_results = list(tqdm(pool.imap(testing_worker, testing_tasks), total=len(testing_tasks), desc="Overall Testing Progress", ncols=100))
    log("Phase 4 Complete.")

    log("Phase 5: Aggregating final results...")
    total_correct, total_preds = 0, 0
    final_cm = {name: Counter() for name in test_data.keys()}
    for true_obj_name, local_cm, correct, total in tqdm(testing_results, desc="Consolidating Results", ncols=100):
        total_correct += correct
        total_preds += total
        final_cm[true_obj_name].update(local_cm)
    log("Phase 5 Complete.")

    accuracy = (total_correct / total_preds) * 100 if total_preds > 0 else 0
    end_time = time.time()
    total_duration = str(datetime.timedelta(seconds=end_time - start_time))

    print("\n" + "="*50)
    log("SIMULATION COMPLETE")
    print("="*50)
    log(f"Total Execution Time: {total_duration}")
    log(f"Final Recognition Accuracy: {accuracy:.2f}%")
    print("="*50)

    serializable_config = {}
    for key, value in vars(config).items():
        if not key.startswith('__') and not isinstance(value, types.ModuleType):
            if isinstance(value, (np.ndarray, jnp.ndarray)):
                serializable_config[key] = np.asarray(value).tolist()
            else:
                serializable_config[key] = value

    metrics = {
        "accuracy": accuracy,
        "correct_predictions": total_correct,
        "total_predictions": total_preds,
        "config": serializable_config,
        "confusion_matrix": {k: dict(v) for k, v in final_cm.items()}
    }
    results_path = config.RESULTS_FILE
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)
    log(f"Save complete.")

# Add the standard multiprocessing guard
if __name__ == "__main__":
    main()