import graphviz
import os
from src import config

def log(message):
    """Prints a message."""
    print(f"[VISUALIZER] {message}")

def visualize_cortex_architecture():
    """
    Generates a diagram of the simulated cortical architecture using Graphviz.
    """
    log("Starting network visualization...")

    dot = graphviz.Digraph('CortexArchitecture', comment='Thousand Brains Model Simulation')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.8', ranksep='1.2')
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='skyblue', fontname='Helvetica')
    dot.attr('edge', color='gray40', arrowsize='0.7')
    dot.attr(label=f"Proposed Network Architecture for M-Brain\n(based on config.py)", fontsize='20')

    with dot.subgraph(name='cluster_cortex') as c:
        c.attr(label='Neocortex Layer', style='rounded', color='lightblue')
        c.node('cortex_center', 'CORTEX\n(Consensus & Aggregation)', shape='octagon', fillcolor='cornflowerblue', fontcolor='white', fontsize='14')

    num_cols_to_draw = min(config.NUM_CORTICAL_COLUMNS, 4)

    for i in range(num_cols_to_draw):
        col_id = f'col_{i}'
        with dot.subgraph(name=f'cluster_col_{i}') as c:
            c.attr(label=f'Column {i+1}', style='rounded,dashed', color='gray50')
            c.node(col_id, f'Cortical Column {i+1}\n(Decentralized Model)', fillcolor='aliceblue')

            dot.edge(col_id, 'cortex_center', style='dashed', label='Votes')

            num_modules_to_draw = min(len(config.GRID_CELL_MODULES), 3)
            for j in range(num_modules_to_draw):
                mod_id = f'mod_{i}_{j}'
                scale = config.GRID_CELL_MODULES[j]['scale']
                dot.node(mod_id, f"Grid Cell Module\nScale: {scale}",
                         shape='ellipse', fillcolor='ivory', fontsize='9')
                dot.edge(mod_id, col_id, label='Location Context')

    with dot.subgraph(name='cluster_legend') as c:
        c.attr(label='Legend', rank='sink')
        c.node('legend1', 'Processing Unit', shape='box', style='rounded,filled', fillcolor='skyblue')
        c.node('legend3', 'Location System Component', shape='ellipse', style='filled', fillcolor='ivory')

    output_path = os.path.join('results', 'cortex_architecture')
    try:
        dot.render(output_path, format='png', view=False, cleanup=True)
        log(f"Network diagram saved to {output_path}.png")
    except Exception as e:
        log(f"Error rendering graph. Please ensure Graphviz is installed and in your system's PATH.")
        log(f"Error: {e}")

if __name__ == '__main__':
    visualize_cortex_architecture()