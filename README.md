# M-Brain: Interactive Simulation

This repository contains an interactive, visual simulation inspired by the Thousand Brains Theory of the neocortex. It demonstrates how a collection of cortical columns can learn feature-location pairs through movement and reach a consensus on object identity through voting.

The simulation uses Pygame to create a graphical interface where you can control an agent and observe its learning and prediction processes in real-time.

![Simulation Screenshot](https://i.imgur.com/your-screenshot-url.png)  ## Features

-   **Interactive Environment**: Control an agent with your keyboard to explore a world with different objects.
-   **Real-time Visualization**: See the agent, objects, and the cortex's votes displayed graphically.
-   **On-the-Fly Learning**: Switch to "Learning Mode" to teach the cortex about new objects.
-   **Robust Location Encoding**: Each cortical column uses a system of multiple grid cell modules for more robust path integration.

## How It Works

-   **ObjectModel**: Stores the association between an object, a specific feature of that object, and its location relative to other features on that object.
-   **GridCellSystem**: Each column tracks its location in the environment using a set of grid cell modules, updating its position based on movement (path integration).
-   **CorticalColumn**: A single column combines a location signal (from its grid cells) with a sensory feature input. In learning mode, it forms an association. In prediction mode, it uses its current location and a sensed feature to predict what object it might be observing.
-   **Cortex**: A collection of columns. When the agent "senses" a feature, every column receives that input. Each column votes for objects that could have that feature at its current location. The votes are aggregated to form a consensus hypothesis about the object's identity.

## Getting Started

### Prerequisites

-   Python 3.8+
-   Pygame and NumPy libraries

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/tgergo1/m-brain.git](https://github.com/tgergo1/m-brain.git)
    cd m-brain
    ```

2.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Simulation

Execute the main simulation script from the root directory of the project:

```bash
python main.py
```

### Controls

-   **Arrow Keys**: Move the agent.
-   **M**: Toggle between **PREDICTING** and **LEARNING** modes.
-   **1, 2, 3**: In **LEARNING** mode, select which object you are teaching the agent about ('Cup', 'Pen', or 'Phone').
-   **Q**: Quit the simulation.