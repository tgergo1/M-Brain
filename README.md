# M-Brain: A Grand-Scale Cortical Simulation

This repository provides a high-performance, parallelized simulation for exploring concepts from the Thousand Brains Theory. This version is optimized for speed using **JAX** for GPU acceleration and `multiprocessing` for parallel execution.

The simulation first trains a distributed cortical model on a procedurally generated dataset of 3D objects. It then evaluates the model's recognition accuracy.

## Core Concepts

* **CorticalColumn**: The fundamental processing unit. Each column is a self-contained agent with its own internal `ObjectModel` and a layered structure.
* **GridCellSystem**: Each column contains its own JAX-accelerated location system, composed of multiple grid cell modules.
* **Cortex**: A collection of thousands of cortical columns that uses a dynamic consensus mechanism for object recognition.
* **Decentralized & Parallel Learning**: Knowledge is learned in parallel by "expert" cortices and then merged into a master cortex for testing.

## Project Workflow

1.  **Compute**: Run the high-performance simulation to train the model and generate performance metrics.
2.  **Analyze**: Use plotting scripts to analyze the results.
3.  **Visualize**: Generate diagrams of the network architecture.

## Installation

### Prerequisites

* Python 3.8+
* Git
* Graphviz (system-level dependency)

### Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/tgergo1/m-brain.git](https://github.com/tgergo1/m-brain.git)
    cd m-brain
    ```

2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install Graphviz Engine:** (See previous instructions)

## Usage Guide

Execute commands from the root directory.

### Step 1: Run the Grand-Scale Simulation

This command starts the main, JAX-accelerated simulation. It will use all available CPU cores and the GPU for training and testing.

```bash
python grand_simulation.py