import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

from src.config import RESULTS_FILE

def plot_confusion_matrix(metrics):
    """Generates and saves a confusion matrix plot."""
    print("Generating confusion matrix plot...")

    cm_data = metrics.get("confusion_matrix", {})
    if not cm_data:
        print("No confusion matrix data found in results.")
        return

    labels = sorted(cm_data.keys())
    cm_array = np.zeros((len(labels), len(labels)), dtype=int)

    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            cm_array[i, j] = cm_data.get(true_label, {}).get(pred_label, 0)

    df_cm = pd.DataFrame(cm_array, index=labels, columns=labels)

    plt.figure(figsize=(12, 10))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='viridis')
    plt.title(f"Confusion Matrix (Accuracy: {metrics.get('accuracy', 0):.2f}%)")
    plt.ylabel('True Object Class')
    plt.xlabel('Predicted Object Class')

    output_path = "results/confusion_matrix.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.show()

def main():
    try:
        with open(RESULTS_FILE, 'r') as f:
            metrics = json.load(f)
    except FileNotFoundError:
        print(f"Error: Results file not found at {RESULTS_FILE}")
        print("Please run 'python grand_simulation.py' first.")
        return

    plot_confusion_matrix(metrics)

if __name__ == "__main__":
    main()