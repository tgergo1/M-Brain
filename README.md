# M-Brain: A Grand-Scale cortical simulation

This repository provides a high-performance, parallelized simulation environment for exploring computational concepts inspired by the Thousand Brains Theory. The project has evolved from a simple interactive demo into a research tool designed for large-scale, data-driven experiments on object recognition and learning.

The simulation first trains a distributed cortical model on a procedurally generated dataset of 3D objects. It then evaluates the model's recognition accuracy and provides a suite of tools for quantitative and architectural analysis. The entire process is designed to leverage multi-core CPU architectures for efficient execution.

## Core concepts

The simulation is built upon a few key components that model the theory's principles:

  * **ObjectModel**: A centralized knowledge base that stores feature-location associations for all learned objects. This acts as the "atlas" that cortical columns reference.
  * **GridCellSystem**: Each cortical column contains its own location system, composed of multiple grid cell modules with unique scales and orientations. This system allows each column to determine its position on an object's surface through path integration.
  * **CorticalColumn**: The fundamental processing unit of the simulation. It receives a sensory feature and, by combining it with its current location, generates a "vote" for the identity of the object being observed.
  * **Cortex**: A collection of thousands of cortical columns that process sensory input in parallel. The Cortex aggregates the votes from all columns to reach a robust consensus on the object's identity, which is its primary recognition mechanism.

## Project workflow

The project is structured as a three-step workflow: **Compute**, **Analyze**, and **Visualize**.

1.  **Compute**: Run the high-performance, parallelized simulation to train the model and generate performance metrics.
2.  **Analyze**: Use plotting scripts to analyze the quantitative results of the simulation, such as accuracy and confusion matrices.
3.  **Visualize**: Generate high-quality diagrams of the simulated network architecture from both a high-level and a neuron-centric perspective.

## Installation

### Prerequisites

  * Python 3.8+
  * Git
  * Graphviz (system-level dependency)

### Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/tgergo1/m-brain.git
    cd m-brain
    ```

2.  **Install Python dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Install the Graphviz Engine:**
    The Python `graphviz` library is a wrapper around the C-based Graphviz engine. You must install the engine separately.

      * **macOS (using Homebrew):**
        ```bash
        brew install graphviz
        ```
      * **Linux (Ubuntu/Debian):**
        ```bash
        sudo apt-get update
        sudo apt-get install graphviz
        ```
      * **Windows:**
        Download an installer from the [official Graphviz website](https://graphviz.org/download/) and ensure its `bin` directory is added to your system's PATH.

## Usage guide

Execute the following commands from the root directory of the project.

### Step 1: Run the grand-scale simulation

This command starts the main simulation. It will use all available CPU cores to generate the dataset, train the cortex, and test its performance. This is the most computationally intensive step.

```bash
python grand_simulation.py
```

Upon completion, this will generate a `results/simulation_metrics.json` file containing all performance data.

### Step 2: Analyze quantitative results

Run the plotting script to generate a confusion matrix from the simulation results.

```bash
python plot_results.py
```

This will display the plot and save it as `results/confusion_matrix.png`.

### Step 3: Visualize the network architectures

You can generate two different views of the simulated network.

  * **High-level architectural diagram:**

    ```bash
    python visualize_network.py
    ```

    This creates a block diagram saved as `results/cortex_architecture.png`.

## Configuration

All key parameters for the simulation can be modified in `src/config.py`. This includes the number of cortical columns, the size of the dataset, the complexity of the grid cell systems, and more, allowing for easy experimentation.