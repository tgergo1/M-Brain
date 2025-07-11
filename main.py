import pygame
import numpy as np
import sys

from src.grid_cells import GridCellModule
from src.object_model import ObjectModel
from src.cortical_column import CorticalColumn
from src.cortex import Cortex
from src.settings import (
    SCREEN_WIDTH, SCREEN_HEIGHT, BG_COLOR, GRID_COLOR, CELL_SIZE,
    AGENT_COLOR, OBJECT_COLORS, FEATURE_COLORS, FONT_SIZE, BLACK, WHITE
)

# --- Object Definitions ---
OBJECTS = {
    "Cup": {
        (2, 2): "handle", (3, 2): "edge", (4, 2): "edge",
        (2, 3): "edge",   (3, 3): "base", (4, 3): "edge",
        (2, 4): "edge",   (3, 4): "base", (4, 4): "edge",
    },
    "Pen": {
        (8, 5): "tip", (8, 6): "body", (8, 7): "body", (8, 8): "body", (8, 9): "end"
    },
    "Phone": {
        (12, 12): "corner", (13, 12): "edge", (14, 12): "edge", (15, 12): "corner",
        (12, 13): "screen", (13, 13): "screen", (14, 13): "screen", (15, 13): "screen",
        (12, 14): "screen", (13, 14): "screen", (14, 14): "screen", (15, 14): "screen",
        (12, 15): "corner", (13, 15): "edge", (14, 15): "edge", (15, 15): "corner",
    }
}

ALL_FEATURES = list(set(feat for obj_feats in OBJECTS.values() for feat in obj_feats.values()))

def get_feature_at_pos(pos):
    """Check if any object has a feature at the given grid position."""
    grid_pos = (pos[0] // CELL_SIZE, pos[1] // CELL_SIZE)
    for obj, features in OBJECTS.items():
        if grid_pos in features:
            return features[grid_pos]
    return None

def build_cortex(num_columns: int = 20) -> Cortex:
    """Builds a cortex with multiple grid cell modules per column."""
    # Start with an empty model for on-the-fly learning
    model = ObjectModel()
    columns = []
    for _ in range(num_columns):
        # Each column gets a more complex set of modules
        modules = [
            GridCellModule(scale=s, orientation=o)
            for s in np.linspace(1.0, 3.0, 4)
            for o in np.linspace(0, np.pi/2, 4)
        ]
        columns.append(CorticalColumn(model, modules))
    return Cortex(columns)

def draw_grid(screen):
    for x in range(0, SCREEN_WIDTH, CELL_SIZE):
        pygame.draw.line(screen, GRID_COLOR, (x, 0), (x, SCREEN_HEIGHT))
    for y in range(0, SCREEN_HEIGHT, CELL_SIZE):
        pygame.draw.line(screen, GRID_COLOR, (0, y), (SCREEN_WIDTH, y))

def draw_objects(screen, font):
    for i, (obj_name, features) in enumerate(OBJECTS.items()):
        obj_color = OBJECT_COLORS[i % len(OBJECT_COLORS)]
        for (gx, gy), feat_name in features.items():
            feat_color = FEATURE_COLORS[ALL_FEATURES.index(feat_name) % len(FEATURE_COLORS)]
            rect = pygame.Rect(gx * CELL_SIZE, gy * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, feat_color, rect)
            pygame.draw.rect(screen, obj_color, rect, 2) # Border

            # Draw feature name
            text_surf = font.render(feat_name[0], True, BLACK)
            text_rect = text_surf.get_rect(center=rect.center)
            screen.blit(text_surf, text_rect)

def draw_agent(screen, agent_pos):
    center_pixel = (agent_pos[0] + CELL_SIZE // 2, agent_pos[1] + CELL_SIZE // 2)
    pygame.draw.circle(screen, AGENT_COLOR, center_pixel, CELL_SIZE // 3)

def draw_ui(screen, font, mode, learning_obj, votes):
    # Mode text
    mode_text = f"Mode: {'LEARNING' if mode == 'learn' else 'PREDICTING'}"
    if mode == 'learn':
        mode_text += f" (Object: {learning_obj})"
    mode_surf = font.render(mode_text, True, BLACK)
    screen.blit(mode_surf, (10, 10))

    # Instructions
    inst_text = "Keys: (M)ode, (1-3) Learn Obj, (Arrows) Move, (Q)uit"
    inst_surf = font.render(inst_text, True, BLACK)
    screen.blit(inst_surf, (10, 35))
    
    # --- Draw Vote Bars ---
    bar_x, bar_y = 10, 70
    bar_height = 20
    max_bar_width = SCREEN_WIDTH - 100
    
    if not votes:
        return

    # Normalize votes for consistent bar length
    total_votes = sum(votes.values())
    if total_votes == 0:
        return
        
    for i, (obj, count) in enumerate(votes.items()):
        # Object name
        obj_surf = font.render(obj, True, BLACK)
        screen.blit(obj_surf, (bar_x, bar_y + i * (bar_height + 5)))

        # Bar
        bar_width = (count / total_votes) * max_bar_width
        bar_rect = pygame.Rect(bar_x + 80, bar_y + i * (bar_height + 5), bar_width, bar_height)
        pygame.draw.rect(screen, OBJECT_COLORS[i % len(OBJECT_COLORS)], bar_rect)
        
        # Vote count
        count_surf = font.render(str(count), True, WHITE)
        screen.blit(count_surf, (bar_x + 85, bar_y + i * (bar_height + 5) + 2))

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("M-Brain Interactive Simulation")
    font = pygame.font.Font(None, FONT_SIZE)
    clock = pygame.time.Clock()

    cortex = build_cortex()
    agent_pos = [SCREEN_WIDTH // 4, SCREEN_HEIGHT // 2]
    movement = np.array([0, 0])

    mode = 'predict'  # 'predict' or 'learn'
    learning_obj_name = list(OBJECTS.keys())[0]
    
    votes = {}

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_m:
                    mode = 'learn' if mode == 'predict' else 'predict'
                # Select object to learn
                if event.key == pygame.K_1:
                    learning_obj_name = list(OBJECTS.keys())[0]
                if event.key == pygame.K_2:
                    learning_obj_name = list(OBJECTS.keys())[1]
                if event.key == pygame.K_3:
                    learning_obj_name = list(OBJECTS.keys())[2]

        # --- Handle Movement ---
        keys = pygame.key.get_pressed()
        movement = np.array([0, 0])
        if keys[pygame.K_LEFT]:
            agent_pos[0] -= CELL_SIZE
            movement[0] = -1
        if keys[pygame.K_RIGHT]:
            agent_pos[0] += CELL_SIZE
            movement[0] = 1
        if keys[pygame.K_UP]:
            agent_pos[1] -= CELL_SIZE
            movement[1] = -1
        if keys[pygame.K_DOWN]:
            agent_pos[1] += CELL_SIZE
            movement[1] = 1

        # Keep agent on screen
        agent_pos[0] = max(0, min(agent_pos[0], SCREEN_WIDTH - CELL_SIZE))
        agent_pos[1] = max(0, min(agent_pos[1], SCREEN_HEIGHT - CELL_SIZE))

        # --- Cortex Interaction ---
        sensed_feature = get_feature_at_pos(agent_pos)
        if sensed_feature and np.any(movement != 0):
            if mode == 'learn':
                # In learning mode, provide the object name
                cortex.sense_and_learn(movement, sensed_feature, learning_obj_name)
                votes = {} # Reset votes when learning
            else: # predict mode
                # In prediction mode, just sense and get votes
                votes = cortex.sense_and_vote(movement, sensed_feature)
        elif not sensed_feature:
            votes = {} # Clear votes if not sensing anything

        # --- Drawing ---
        screen.fill(BG_COLOR)
        draw_grid(screen)
        draw_objects(screen, font)
        draw_agent(screen, agent_pos)
        draw_ui(screen, font, mode, learning_obj_name, votes)

        pygame.display.flip()
        clock.tick(10) # Limit frame rate

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()