import numpy as np
import json
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
from collections import Counter

from src import config
from src.dataset_generator import generate_dataset
from src.grid_cells import GridCellModule
from src.object_model import ObjectModel
from src.cortical_column import CorticalColumn
from src.cortex import Cortex


def create_rotation_matrix(angles):
    """Creates a 3x3 rotation matrix from Euler angles (roll, pitch, yaw)."""
    return R.from_euler('xyz', angles, degrees=False).as_matrix()

def build_cortex_from_config() -> Cortex:
    """Builds the cortex based on global configuration."""
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

def generate_sensory_sequence(obj_features, num_steps, move_std_dev):
    """OPTIMIZED: Simulates a sensor moving over an object's surface using a k-d tree."""
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
        distance, index = kdtree.query(current_pos)
        closest_feature_loc = feature_locations[index]
        features.append(obj_features[closest_feature_loc])
        
    return movements, features

def main():
    print("--- Grand Scale M-Brain Simulation ---")
    
    # 1. Build Cortex
    print(f"Building cortex with {config.NUM_CORTICAL_COLUMNS} columns...")
    cortex = build_cortex_from_config()

    # 2. Generate Datasets
    print("Generating datasets...")
    train_data = generate_dataset(
        config.NUM_OBJECT_TYPES, config.FEATURES_PER_OBJECT, config.DATASET_SIZE_TRAIN, desc="Train"
    )
    test_data = generate_dataset(
        config.NUM_OBJECT_TYPES, config.FEATURES_PER_OBJECT, config.DATASET_SIZE_TEST, desc="Test"
    )

    # 3. Training Phase
    print("\nStarting training phase...")
    # Outer loop for object classes
    for obj_name, instances in tqdm(train_data.items(), desc="Training on object classes"):
        # --- FIX: Add inner progress bar for instances ---
        # This will show progress within each class, preventing the appearance of being stuck.
        # `leave=False` makes this bar disappear after it's done, keeping the output clean.
        for instance_features in tqdm(instances, desc=f"  ↳ Instances of {obj_name}", leave=False):
            movements, features = generate_sensory_sequence(
                instance_features, config.SENSORY_STEPS_PER_OBJECT, config.MOVEMENT_STD_DEV
            )
            cortex.process_sensory_sequence(movements, features, learn=True, obj_name=obj_name)
    
    # 4. Testing Phase
    print("\nStarting testing phase...")
    correct_predictions = 0
    total_predictions = 0
    confusion_matrix = {name: Counter() for name in test_data.keys()}

    for true_obj_name, instances in tqdm(test_data.items(), desc="Testing object classes"):
        for instance_features in tqdm(instances, desc=f"  ↳ Instances of {true_obj_name}", leave=False):
            movements, features = generate_sensory_sequence(
                instance_features, config.SENSORY_STEPS_PER_OBJECT, config.MOVEMENT_STD_DEV
            )
            votes = cortex.process_sensory_sequence(movements, features, learn=False)
            
            if votes:
                predicted_obj_name = votes.most_common(1)[0][0]
                confusion_matrix[true_obj_name].update([predicted_obj_name])
                if predicted_obj_name == true_obj_name:
                    correct_predictions += 1
            total_predictions += 1
            
    # 5. Log Results
    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    print(f"\n--- Simulation Complete ---")
    print(f"Final Recognition Accuracy: {accuracy:.2f}%")

    serializable_config = {k: v for k, v in vars(config).items() if not k.startswith('__')}
    for k, v in serializable_config.items():
        if isinstance(v, np.ndarray):
            serializable_config[k] = v.tolist()

    metrics = {
        "accuracy": accuracy,
        "correct_predictions": correct_predictions,
        "total_predictions": total_predictions,
        "config": serializable_config,
        "confusion_matrix": {k: dict(v) for k, v in confusion_matrix.items()}
    }

    print(f"Saving results to {config.RESULTS_FILE}...")
    with open(config.RESULTS_FILE, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print("\nTo visualize results, run: python plot_results.py")


if __name__ == "__main__":
    main()