import pygame
import numpy as np
import sys

from src.grid_cells import GridCellModule
from src.cortical_column import CorticalColumn
from src.cortex import Cortex
from src.settings import (
    SCREEN_WIDTH, SCREEN_HEIGHT, BG_COLOR, GRID_COLOR, CELL_SIZE,
    AGENT_COLOR, OBJECT_COLORS, FEATURE_COLORS, FONT_SIZE, BLACK, WHITE
)

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
    """Builds a cortex with layered columns for the interactive demo."""
    columns = []
    for _ in range(num_columns):
        modules = [
            GridCellModule(scale=s, orientation=np.identity(3))
            for s in np.linspace(20.0, 50.0, 4)
        ]
        columns.append(CorticalColumn(modules))
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
            pygame.draw.rect(screen, obj_color, rect, 2)

            text_surf = font.render(feat_name[0], True, BLACK)
            text_rect = text_surf.get_rect(center=rect.center)
            screen.blit(text_surf, text_rect)

def draw_agent(screen, agent_pos):
    center_pixel = (agent_pos[0] + CELL_SIZE // 2, agent_pos[1] + CELL_SIZE // 2)
    pygame.draw.circle(screen, AGENT_COLOR, center_pixel, CELL_SIZE // 3)

def draw_ui(screen, font, mode, learning_obj, votes):
    mode_text = f"Mode: {'LEARNING' if mode == 'learn' else 'PREDICTING'}"
    if mode == 'learn':
        mode_text += f" (Object: {learning_obj})"
    mode_surf = font.render(mode_text, True, BLACK)
    screen.blit(mode_surf, (10, 10))

    inst_text = "Keys: (M)ode, (1-3) Learn Obj, (Arrows) Move, (Q)uit"
    inst_surf = font.render(inst_text, True, BLACK)
    screen.blit(inst_surf, (10, 35))

    bar_x, bar_y = 10, 70
    bar_height = 20
    max_bar_width = SCREEN_WIDTH - 120

    if not votes:
        return

    total_votes = sum(votes.values())
    if total_votes == 0:
        return

    sorted_votes = sorted(votes.items(), key=lambda item: item[1], reverse=True)

    for i, (obj, count) in enumerate(sorted_votes):
        obj_surf = font.render(obj, True, BLACK)
        screen.blit(obj_surf, (bar_x, bar_y + i * (bar_height + 5)))

        bar_width = (count / total_votes) * max_bar_width
        bar_rect = pygame.Rect(bar_x + 90, bar_y + i * (bar_height + 5), bar_width, bar_height)
        try:
            obj_index = list(OBJECTS.keys()).index(obj)
            color = OBJECT_COLORS[obj_index % len(OBJECT_COLORS)]
        except ValueError:
            color = (128, 128, 128)
        pygame.draw.rect(screen, color, bar_rect)

        count_surf = font.render(str(count), True, WHITE)
        screen.blit(count_surf, (bar_x + 95, bar_y + i * (bar_height + 5) + 2))

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("M-Brain Interactive Simulation (Layered Architecture)")
    font = pygame.font.Font(None, FONT_SIZE)
    clock = pygame.time.Clock()

    cortex = build_cortex()
    agent_pos = [SCREEN_WIDTH // 4, SCREEN_HEIGHT // 2]

    mode = 'predict'
    learning_obj_name = list(OBJECTS.keys())[0]
    votes = Counter()

    running = True
    while running:
        movement_2d = np.array([0, 0])
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    mode = 'learn' if mode == 'predict' else 'predict'
                    votes.clear()
                if event.key == pygame.K_1: learning_obj_name = list(OBJECTS.keys())[0]
                if event.key == pygame.K_2: learning_obj_name = list(OBJECTS.keys())[1]
                if event.key == pygame.K_3: learning_obj_name = list(OBJECTS.keys())[2]

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:  movement_2d[0] = -1
        if keys[pygame.K_RIGHT]: movement_2d[0] = 1
        if keys[pygame.K_UP]:    movement_2d[1] = -1
        if keys[pygame.K_DOWN]:  movement_2d[1] = 1

        if np.any(movement_2d != 0):
            agent_pos[0] += movement_2d[0] * CELL_SIZE
            agent_pos[1] += movement_2d[1] * CELL_SIZE
            agent_pos[0] = max(0, min(agent_pos[0], SCREEN_WIDTH - CELL_SIZE))
            agent_pos[1] = max(0, min(agent_pos[1], SCREEN_HEIGHT - CELL_SIZE))

            sensed_feature = get_feature_at_pos(agent_pos)
            movement_3d = np.array([movement_2d[0] * CELL_SIZE, movement_2d[1] * CELL_SIZE, 0.0])

            if mode == 'learn':
                if sensed_feature:
                    cortex.process_sensory_sequence([movement_3d], [sensed_feature], learn=True, obj_name=learning_obj_name)
                votes.clear()
            else:
                if sensed_feature:
                    votes = cortex.process_sensory_sequence([movement_3d], [sensed_feature], learn=False)
                else:
                    votes.clear()

        screen.fill(BG_COLOR)
        draw_grid(screen)
        draw_objects(screen, font)
        draw_agent(screen, agent_pos)
        draw_ui(screen, font, mode, learning_obj_name, votes)

        pygame.display.flip()
        clock.tick(10)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()