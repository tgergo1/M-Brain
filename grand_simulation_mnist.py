import jax
import jax.numpy as jnp
from jax import random
from jax import config as jax_config
import numpy as np
import multiprocessing as mp
import time
import datetime
import os
import argparse
import pickle
from tqdm import tqdm
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Import project modules
from src import config_mnist as config
from src.mnist_loader import load_mnist_data
from src.grid_cells import GridCellModule
from src.cortical_column import CorticalColumn
from src.cortex import Cortex

# --- Configuration & Setup ---
jax_config.update("jax_enable_x64", True)

def log(message):
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def build_cortex_from_config() -> Cortex:
    columns = [CorticalColumn([GridCellModule(scale=m['scale']) for m in config.GRID_CELL_MODULES]) for _ in range(config.NUM_CORTICAL_COLUMNS)]
    return Cortex(columns)

def generate_mnist_sensory_sequence(image, num_steps, patch_size, key):
    digit_coords = jnp.argwhere(image > 0)
    if digit_coords.shape[0] == 0:
        return [], [], [], key

    key, subkey = random.split(key)
    glimpse_indices = random.choice(subkey, digit_coords.shape[0], shape=(num_steps,), replace=True)
    glimpse_centers = digit_coords[glimpse_indices]

    movements, features, locations = [], [], []
    last_center = glimpse_centers[0]

    for center in glimpse_centers:
        movement = center - last_center
        movements.append(jnp.array([movement[0], movement[1], 0.0]))
        locations.append(tuple(np.asarray(last_center).astype(float)))
        last_center = center

        half_patch = patch_size // 2
        x, y = center
        patch = jax.lax.dynamic_slice(image, (x - half_patch, y - half_patch), (patch_size, patch_size))
        feature_str = f"p-{''.join(map(str, patch.flatten().astype(int)))}"
        features.append(feature_str)
        
    return movements, features, locations, key

# --- Plotting Process ---
def live_plotter(queue: mp.Queue, num_workers: int, total_steps: int):
    """
    This function runs in its own process.
    It reads accuracy data from a queue and updates a plot in real-time.
    """
    accuracies = [[] for _ in range(num_workers)]
    steps = [[] for _ in range(num_workers)]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    lines = [ax.plot([], [], label=f'Worker {i+1}', alpha=0.8)[0] for i in range(num_workers)]
    avg_line, = ax.plot([], [], 'k-', label='Average Accuracy', linewidth=2.5)
    
    all_data_points = []

    def init():
        ax.set_xlim(0, total_steps)
        ax.set_ylim(0, 100)
        ax.set_title('Live Training Accuracy', fontsize=16, fontweight='bold')
        ax.set_xlabel('Training Step (per worker)', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.legend(loc='lower right')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        fig.tight_layout()
        return lines + [avg_line]

    def update(frame):
        while not queue.empty():
            data = queue.get()
            if data is None: # Sentinel value to stop
                return lines + [avg_line]
            worker_id, step, acc = data
            all_data_points.append((step, acc))
            accuracies[worker_id].append(acc)
            steps[worker_id].append(step)
            lines[worker_id].set_data(steps[worker_id], accuracies[worker_id])
            
            # Update average line
            if len(all_data_points) > 1:
                sorted_points = sorted(all_data_points)
                x_points, y_points = zip(*sorted_points)
                # Apply a rolling average to smooth the curve
                y_smooth = pd.Series(y_points).rolling(window=max(1, len(y_points)//10), min_periods=1).mean()
                avg_line.set_data(x_points, y_smooth)

        return lines + [avg_line]

    # Use FuncAnimation for efficient real-time plotting
    _ = FuncAnimation(fig, update, init_func=init, blit=True, interval=500)
    plt.show()

# --- Worker Functions ---
def training_worker(args):
    position, images, labels, accuracy_queue, seed = args
    worker_id = position - 1
    pid = os.getpid()
    log(f"[Worker PID: {pid}] Starting training for worker {worker_id+1}")
    
    key = random.PRNGKey(seed)
    cortex = build_cortex_from_config()
    
    correct_predictions, total_predictions = 0, 0

    for i, (image, label) in enumerate(tqdm(zip(images, labels), desc=f"  ↳ Worker {worker_id+1}", total=len(images), position=position, leave=False, ncols=110)):
        key, subkey = random.split(key)
        movements, features, locations, key = generate_mnist_sensory_sequence(
            image, config.SENSORY_STEPS_PER_OBJECT, config.PATCH_SIZE, subkey
        )
        
        true_label_str = str(label)
        
        votes, active_columns = cortex.process_sensory_sequence(movements, features, learn=False, return_active_columns=True)
        
        if votes:
            predicted_obj_name = votes.most_common(1)[0][0]
            total_predictions += 1
            if predicted_obj_name == true_label_str:
                correct_predictions += 1
            else:
                culprit_columns = active_columns.get(predicted_obj_name, [])
                cortex.apply_targeted_feedback(locations, features, predicted_obj_name, culprit_columns, -config.UNLEARNING_RATE)
                
                correct_columns = active_columns.get(true_label_str, [])
                cortex.apply_targeted_feedback(locations, features, true_label_str, config.REINFORCEMENT_RATE)
        
        cortex.process_sensory_sequence(movements, features, learn=True, obj_name=true_label_str)

        if (i + 1) % config.LOG_FREQUENCY == 0 and total_predictions > 0:
            accuracy = (correct_predictions / total_predictions) * 100
            accuracy_queue.put((worker_id, i + 1, accuracy))
            correct_predictions, total_predictions = 0, 0

    log(f"[Worker PID: {pid}] Finished training for worker {worker_id+1}.")
    return [col.layer2_3.object_model.storage for col in cortex.columns]

def testing_worker(args):
    # This function remains unchanged
    position, images, true_labels, all_learned_data, seed = args
    pid = os.getpid()
    log(f"[Worker PID: {pid}] Rebuilding cortex and testing...")

    master_cortex = build_cortex_from_config()
    for i, column_data in enumerate(all_learned_data):
        master_cortex.columns[i].layer2_3.object_model.storage = column_data
    
    key = random.PRNGKey(seed)
    local_confusion_matrix = defaultdict(Counter)
    correct_count, total_count = 0, len(images)

    for image, true_label in tqdm(zip(images, true_labels), desc=f"  ↳ PID {pid} Testing", total=len(images), position=position, leave=False, ncols=100):
        key, subkey = random.split(key)
        _, features, _, key = generate_mnist_sensory_sequence(image, config.SENSORY_STEPS_PER_OBJECT, config.PATCH_SIZE, subkey)
        movements = [jnp.zeros(3) for _ in features]
        votes = master_cortex.process_sensory_sequence(movements, features, learn=False)
        if votes:
            predicted_obj_name = votes.most_common(1)[0][0]
            local_confusion_matrix[str(true_label)].update([predicted_obj_name])
            if predicted_obj_name == str(true_label):
                correct_count += 1
    return local_confusion_matrix, correct_count, total_count

# --- Main Execution ---
def run_training(start_time):
    log("Mode: Training")
    log("Phase 1: Loading MNIST dataset...")
    train_images, train_labels, _, _ = load_mnist_data(train_limit=config.DATASET_SIZE_TRAIN)
    log("Phase 1 Complete.")

    num_processes = min(mp.cpu_count(), 10)
    manager = mp.Manager()
    accuracy_queue = manager.Queue()

    train_splits = np.array_split(np.arange(len(train_images)), num_processes)
    max_steps_per_worker = max(len(s) for s in train_splits)

    plot_process = mp.Process(target=live_plotter, args=(accuracy_queue, num_processes, max_steps_per_worker))
    plot_process.start()

    log("Phase 2: Starting Parallel Training...")
    training_tasks = [(i + 1, train_images[split], train_labels[split], accuracy_queue, int(start_time) + i) for i, split in enumerate(train_splits)]

    with mp.Pool(processes=num_processes) as pool:
        training_results = pool.map(training_worker, training_tasks)
    log("Phase 2 Complete.")
    
    # Signal the plotting process to stop
    accuracy_queue.put(None)
    time.sleep(2) # Give plotter time to receive signal
    plot_process.terminate()
    plot_process.join()

    log("Phase 3: Aggregating models...")
    master_learned_data = [defaultdict(list) for _ in range(config.NUM_CORTICAL_COLUMNS)]
    for expert_data in tqdm(training_results, desc="Aggregating Models"):
        for i, column_data in enumerate(expert_data):
            for key, value in column_data.items():
                master_learned_data[i][key].extend(value)
    log("Phase 3 Complete.")
    
    log("Phase 4: Saving model...")
    model_path = config.MODEL_FILE
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(master_learned_data, f)
    log(f"Model saved to {model_path}")

def run_testing(start_time):
    # This function remains unchanged...
    log("Mode: Testing")
    log("Phase 1: Loading test data and saved model...")
    model_path = config.MODEL_FILE
    if not os.path.exists(model_path):
        log(f"Error: Model file not found at {model_path}. Please run in 'train' mode first.")
        return

    _, _, test_images, test_labels = load_mnist_data(test_limit=config.DATASET_SIZE_TEST)
    with open(model_path, 'rb') as f:
        master_learned_data = pickle.load(f)
    log("Phase 1 Complete.")
    
    log("Phase 2: Parallel Testing...")
    num_processes = min(mp.cpu_count(), 10)
    log(f"Spawning {num_processes} workers...")
    test_splits = np.array_split(np.arange(len(test_images)), num_processes)
    testing_tasks = [(i + 1, test_images[split], test_labels[split], master_learned_data, int(start_time) + 1000 + i) for i, split in enumerate(test_splits)]

    with mp.Pool(processes=num_processes) as pool:
        testing_results = list(tqdm(pool.imap(testing_worker, testing_tasks), total=len(testing_tasks), desc="Overall Testing"))
    log("Phase 2 Complete.")

    log("Phase 3: Aggregating final results...")
    total_correct, total_preds = 0, 0
    for _, correct, total in tqdm(testing_results, desc="Consolidating"):
        total_correct += correct
        total_preds += total
    log("Phase 3 Complete.")

    accuracy = (total_correct / total_preds) * 100 if total_preds > 0 else 0
    log(f"Final Recognition Accuracy: {accuracy:.2f}% ({total_correct}/{total_preds})")

def main():
    parser = argparse.ArgumentParser(description="M-Brain Final MNIST Simulation with Live Plotting.")
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True,
                        help="Set the mode to 'train' a new model or 'test' an existing one.")
    args = parser.parse_args()
    
    start_time = time.time()
    log("--- M-Brain MNIST Simulation (Live Plotting) ---")

    if args.mode == 'train':
        run_training(start_time)
    elif args.mode == 'test':
        run_testing(start_time)
        
    log(f"Total Execution Time: {str(datetime.timedelta(seconds=time.time() - start_time))}")

if __name__ == "__main__":
    # This guard is essential for multiprocessing to work correctly.
    mp.set_start_method('spawn', force=True)
    main()