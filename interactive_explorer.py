import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D
import textwrap

# Use a specific results file path to avoid circular dependency
RESULTS_FILE_PATH = "results/simulation_metrics.json"

class ResultsExplorer:
    def __init__(self, metrics, dataset):
        self.metrics = metrics
        self.dataset = dataset
        self.object_names = sorted(self.dataset.keys())
        self.current_obj_index = 0

        # --- Create the main figure and subplots ---
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.canvas.manager.set_window_title('M-Brain Interactive Results Explorer')
        
        # Define layout using gridspec for more control
        gs = self.fig.add_gridspec(3, 3)
        self.ax_3d = self.fig.add_subplot(gs[:, 0], projection='3d')
        self.ax_info = self.fig.add_subplot(gs[0, 1])
        self.ax_cm = self.fig.add_subplot(gs[1:, 1:])

        # --- Navigation Buttons ---
        self.ax_prev = plt.axes([0.35, 0.05, 0.1, 0.075])
        self.ax_next = plt.axes([0.46, 0.05, 0.1, 0.075])
        self.btn_prev = Button(self.ax_prev, 'Previous Object')
        self.btn_next = Button(self.ax_next, 'Next Object')
        self.btn_prev.on_clicked(self.prev_object)
        self.btn_next.on_clicked(self.next_object)
        
        # --- Initial Draws ---
        self.draw_info_panel()
        self.draw_confusion_matrix()
        self.update_3d_plot()

    def draw_info_panel(self):
        """Displays key parameters and overall accuracy."""
        self.ax_info.clear()
        self.ax_info.set_title("Simulation Summary", fontsize=14)
        self.ax_info.axis('off')

        accuracy = self.metrics.get('accuracy', 0)
        config = self.metrics.get('config', {})
        
        info_text = (
            f"Overall Accuracy: {accuracy:.2f}%\n"
            f"\n--- Cortex Configuration ---\n"
            f"Columns: {config.get('NUM_CORTICAL_COLUMNS', 'N/A')}\n"
            f"Modules per Column: {len(config.get('GRID_CELL_MODULES', []))}\n"
            f"\n--- Dataset ---\n"
            f"Object Classes: {config.get('NUM_OBJECT_TYPES', 'N/A')}\n"
            f"Training Instances: {config.get('DATASET_SIZE_TRAIN', 'N/A')}\n"
            f"Testing Instances: {config.get('DATASET_SIZE_TEST', 'N/A')}"
        )
        
        self.ax_info.text(0.05, 0.95, info_text, transform=self.ax_info.transAxes,
                          verticalalignment='top', family='monospace', fontsize=10)

    def draw_confusion_matrix(self):
        """Displays the confusion matrix as a heatmap."""
        self.ax_cm.clear()
        cm_data = self.metrics.get("confusion_matrix", {})
        labels = sorted(cm_data.keys())
        
        if not labels:
            self.ax_cm.text(0.5, 0.5, "No Confusion Matrix Data", ha='center', va='center')
            return

        cm_array = np.array([[cm_data.get(r, {}).get(c, 0) for c in labels] for r in labels])
        
        im = self.ax_cm.imshow(cm_array, cmap="viridis")
        self.ax_cm.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right")
        self.ax_cm.set_yticks(np.arange(len(labels)), labels=labels)
        
        # Loop over data dimensions and create text annotations.
        for i in range(len(labels)):
            for j in range(len(labels)):
                self.ax_cm.text(j, i, cm_array[i, j], ha="center", va="center", color="w")

        self.ax_cm.set_title("Recognition Confusion Matrix", fontsize=14)
        self.fig.colorbar(im, ax=self.ax_cm, fraction=0.046, pad=0.04)

    def update_3d_plot(self):
        """Updates the 3D scatter plot to show the current object."""
        self.ax_3d.clear()
        obj_name = self.object_names[self.current_obj_index]
        # We visualize the first instance of the object class
        obj_instance = self.dataset[obj_name][0]
        
        locations = np.array(list(obj_instance.keys()))
        if locations.size == 0:
            return

        x, y, z = locations[:, 0], locations[:, 1], locations[:, 2]
        
        self.ax_3d.scatter(x, y, z, c=z, cmap='inferno', marker='o')
        self.ax_3d.set_title(f"Feature Map:\n{obj_name}", fontsize=14)
        self.ax_3d.set_xlabel("X coordinate")
        self.ax_3d.set_ylabel("Y coordinate")
        self.ax_3d.set_zlabel("Z coordinate")
        self.fig.canvas.draw_idle()

    def next_object(self, event):
        """Callback to show the next object in the 3D plot."""
        self.current_obj_index = (self.current_obj_index + 1) % len(self.object_names)
        self.update_3d_plot()

    def prev_object(self, event):
        """Callback to show the previous object in the 3D plot."""
        self.current_obj_index = (self.current_obj_index - 1) % len(self.object_names)
        self.update_3d_plot()

def main():
    try:
        with open(RESULTS_FILE_PATH, 'r') as f:
            metrics = json.load(f)
    except FileNotFoundError:
        print(f"Error: Results file not found at {RESULTS_FILE_PATH}")
        print("Please run 'python grand_simulation.py' first.")
        return

    # To visualize the objects, we need to generate one instance of the dataset
    # This is lightweight and avoids saving the entire large dataset to disk
    print("Generating a sample dataset for visualization...")
    from src.dataset_generator import generate_dataset
    config = metrics.get('config', {})
    sample_dataset = generate_dataset(
        config.get('NUM_OBJECT_TYPES', 10),
        config.get('FEATURES_PER_OBJECT', 50),
        1, # Only need one instance per class for visualization
        desc="Sample Viz"
    )

    explorer = ResultsExplorer(metrics, sample_dataset)
    plt.tight_layout(rect=[0, 0.1, 1, 0.95]) # Adjust layout to prevent overlap
    plt.show()

if __name__ == "__main__":
    main()