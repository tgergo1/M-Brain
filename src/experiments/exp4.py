import asyncio
import math
import random
import pygame
from pygame.locals import *
import numpy as np
import time

# --- Configuration Constants ---
# General
SCREEN_WIDTH = 1600
SCREEN_HEIGHT = 900
FPS = 60
BACKGROUND_COLOR = (10, 10, 20)
FONT_COLOR = (200, 200, 220)

# Neuron
NEURON_MIN_DENDRITES = 2
NEURON_MAX_DENDRITES = 4
NEURON_BASE_ENERGY_COST = 0.001 
NEURON_SHRINK_THRESHOLD = 0.1
NEURON_STAGNATE_ENERGY_LEVEL = 5.0
NEURON_AWAKEN_ENERGY_LEVEL = 15.0
NEURON_PUFFER_CAPACITY = 100.0
NEURON_EMIT_CHARGE_AMOUNT = 120.0
NEURON_EMIT_ENERGY_COST = 5.0
NEURON_CONNECTION_RADIUS = 15.0 

# Cortex
CORTEX_ENERGY_CAPACITY = 100000.0
CORTEX_ENERGY_REGEN_RATE = 10.0
CORTEX_REWARD_ENERGY = 5000.0

# Visualization
CAMERA_SPEED = 20.0
CAMERA_ROTATION_SPEED = 0.02
CONNECTION_COLOR = (60, 90, 150)
ACTIVE_CONNECTION_COLOR = (150, 200, 255)
SPHERE_COLOR = (255, 100, 100, 50) # RGBA

# --- Helper Functions & Classes ---

class Vector3:
    """A simple 3D vector class."""
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z

    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar):
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalize(self):
        mag = self.magnitude()
        if mag == 0: return Vector3(0, 0, 0)
        return Vector3(self.x / mag, self.y / mag, self.z / mag)

    def distance_to(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)

    def to_tuple(self):
        return (self.x, self.y, self.z)

class Camera:
    """A 3D camera for the visualization."""
    def __init__(self, pos=Vector3(0, 0, -500), target=Vector3(0, 0, 0)):
        self.pos = pos
        self.target = target
        self.up = Vector3(0, 1, 0)
        self.yaw = -math.pi / 2
        self.pitch = 0
        self.update_vectors()

    def update_vectors(self):
        front_x = math.cos(self.yaw) * math.cos(self.pitch)
        front_y = math.sin(self.pitch)
        front_z = math.sin(self.yaw) * math.cos(self.pitch)
        self.front = Vector3(front_x, front_y, front_z).normalize()
        self.right = self.front.normalize().__class__(self.front.z, 0, -self.front.x).normalize()
        self.up = self.right.normalize().__class__(
            self.right.y * self.front.z - self.right.z * self.front.y,
            self.right.z * self.front.x - self.right.x * self.front.z,
            self.right.x * self.front.y - self.right.y * self.front.x
        ).normalize()


    def get_view_matrix(self):
        # This is a simplified lookAt matrix calculation
        z_axis = (self.pos - self.target).normalize()
        x_axis = self.up.normalize().__class__(self.up.y * z_axis.z - self.up.z * z_axis.y, self.up.z * z_axis.x - self.up.x * z_axis.z, self.up.x * z_axis.y - self.up.y * z_axis.x).normalize()
        y_axis = z_axis.normalize().__class__(z_axis.y * x_axis.z - z_axis.z * x_axis.y, z_axis.z * x_axis.x - z_axis.x * x_axis.z, z_axis.x * x_axis.y - z_axis.y * x_axis.x)
        
        translation = np.identity(4)
        translation[3, 0] = -self.pos.x
        translation[3, 1] = -self.pos.y
        translation[3, 2] = -self.pos.z
        
        rotation = np.identity(4)
        rotation[0, 0] = x_axis.x
        rotation[1, 0] = x_axis.y
        rotation[2, 0] = x_axis.z
        rotation[0, 1] = y_axis.x
        rotation[1, 1] = y_axis.y
        rotation[2, 1] = y_axis.z
        rotation[0, 2] = z_axis.x
        rotation[1, 2] = z_axis.y
        rotation[2, 2] = z_axis.z

        return np.dot(translation, rotation)

    def process_input(self, keys, mouse_rel):
        if keys[K_w]: self.pos += self.front * CAMERA_SPEED
        if keys[K_s]: self.pos -= self.front * CAMERA_SPEED
        if keys[K_a]: self.pos -= self.right * CAMERA_SPEED
        if keys[K_d]: self.pos += self.right * CAMERA_SPEED
        if keys[K_SPACE]: self.pos += self.up * CAMERA_SPEED
        if keys[K_LSHIFT]: self.pos -= self.up * CAMERA_SPEED

        dx, dy = mouse_rel
        self.yaw += dx * CAMERA_ROTATION_SPEED
        self.pitch -= dy * CAMERA_ROTATION_SPEED
        self.pitch = max(-math.pi/2 + 0.01, min(math.pi/2 - 0.01, self.pitch))
        self.update_vectors()
        self.target = self.pos + self.front

# --- Core Simulation Classes ---

class Neuron:
    """Represents a single neuron in the 3D space."""
    def __init__(self, cortex, position: Vector3):
        self.cortex = cortex
        self.id = id(self)
        self.position = position
        self.dendrites = [Dendrite(self) for _ in range(random.randint(NEURON_MIN_DENDRITES, NEURON_MAX_DENDRITES))]
        self.axon = Axon(self)
        self.puffer = 0.0
        self.energy = 20.0
        self.is_stagnant = False
        self.last_activity_time = time.time()
        self.growth_sphere_radius = 0
        self.growth_sphere_active = False

    async def run(self):
        """The main lifecycle loop for the neuron."""
        while True:
            await self.consume_energy()
            await self.check_stagnation()
            
            if not self.is_stagnant:
                if self.puffer >= NEURON_PUFFER_CAPACITY:
                    await self.emit_charge()
                    self.puffer = 0.0
            
            if self.growth_sphere_active:
                self.growth_sphere_radius += 0.5 # Grow sphere
                if self.growth_sphere_radius > 50: # Reset if no connection found
                    self.growth_sphere_active = False
                    self.growth_sphere_radius = 0

            await asyncio.sleep(0.01) # Neuron's internal clock tick

    async def consume_energy(self):
        """Neuron consumes energy from the cortex just to exist."""
        cost = NEURON_BASE_ENERGY_COST
        drawn_energy = self.cortex.request_energy(self.position, cost)
        self.energy += drawn_energy - cost
        if self.energy < 0: self.energy = 0

    async def check_stagnation(self):
        """Check if the neuron should become stagnant or awaken."""
        if self.energy < NEURON_STAGNATE_ENERGY_LEVEL and not self.is_stagnant:
            # Check if it has been low on energy for a while
            if time.time() - self.last_activity_time > 5.0:
                self.is_stagnant = True
                await self.go_stagnant()
        elif self.energy > NEURON_AWAKEN_ENERGY_LEVEL and self.is_stagnant:
            self.is_stagnant = False
            self.last_activity_time = time.time()

    async def go_stagnant(self):
        """Disconnects all interfaces and enters a low-power state."""
        print(f"Neuron {self.id} is going stagnant.")
        await self.axon.disconnect()
        for dendrite in self.dendrites:
            await dendrite.disconnect()
        self.puffer = 0
        
    async def receive_charge(self, amount: float, source_axon):
        """Receives charge from another neuron's axon."""
        if self.is_stagnant:
            return
        self.puffer += amount
        self.energy += amount * 0.1 # Small energy gain from receiving charge
        self.last_activity_time = time.time()
        # Visual feedback
        source_axon.last_emit_time = time.time()


    async def emit_charge(self):
        """Emits charge from the axon if connected."""
        if self.is_stagnant:
            return

        cost = NEURON_EMIT_ENERGY_COST
        available_energy = self.cortex.request_energy(self.position, cost)
        self.energy += available_energy

        if self.energy < cost:
            # Not enough energy to emit fully
            emit_factor = self.energy / cost
            charge_to_emit = NEURON_EMIT_CHARGE_AMOUNT * emit_factor
            self.energy = 0
        else:
            self.energy -= cost
            charge_to_emit = NEURON_EMIT_CHARGE_AMOUNT
        
        if charge_to_emit > 0:
            self.last_activity_time = time.time()
            if self.axon.is_connected():
                await self.axon.emit(charge_to_emit)
            else:
                # If not connected, grow a virtual sphere
                self.growth_sphere_active = True
                self.growth_sphere_radius = 1.0
                await self.cortex.find_connection_for(self)

class Dendrite:
    """Receiving interface for a Neuron."""
    def __init__(self, neuron: Neuron):
        self.neuron = neuron
        self.connection = None # Connected Axon

    def is_connected(self):
        return self.connection is not None

    async def connect(self, axon):
        if not self.is_connected():
            self.connection = axon
            return True
        return False

    async def disconnect(self):
        if self.is_connected():
            axon = self.connection
            self.connection = None
            if axon.connection == self:
                await axon.disconnect() # Ensure bidirectional disconnect

    async def receive(self, amount: float):
        await self.neuron.receive_charge(amount, self.connection)

class Axon:
    """Emitting interface for a Neuron."""
    def __init__(self, neuron: Neuron):
        self.neuron = neuron
        self.connection = None # Connected Dendrite
        self.last_emit_time = -1

    def is_connected(self):
        return self.connection is not None

    async def connect(self, dendrite: Dendrite):
        if not self.is_connected():
            self.connection = dendrite
            await dendrite.connect(self) # Ensure bidirectional connection
            self.neuron.growth_sphere_active = False
            self.neuron.growth_sphere_radius = 0
            return True
        return False

    async def disconnect(self):
        if self.is_connected():
            dendrite = self.connection
            self.connection = None
            if dendrite.connection == self:
                await dendrite.disconnect() # Ensure bidirectional disconnect

    async def emit(self, amount: float):
        if self.is_connected():
            self.last_emit_time = time.time()
            await self.connection.receive(amount)

class Cortex:
    """A 3D cuboid space containing neurons and managing energy."""
    def __init__(self, position: Vector3, size: Vector3, neuron_count: int, is_input=False, is_output=False):
        self.position = position
        self.size = size
        self.neurons = []
        self.energy_level = CORTEX_ENERGY_CAPACITY
        self.is_input = is_input
        self.is_output = is_output
        self.input_interface = []
        self.output_interface = []
        
        self.populate_neurons(neuron_count)
        self.setup_interfaces()

    def populate_neurons(self, count):
        """Create neurons within the cortex volume."""
        for _ in range(count):
            pos = Vector3(
                self.position.x + random.uniform(-self.size.x / 2, self.size.x / 2),
                self.position.y + random.uniform(-self.size.y / 2, self.size.y / 2),
                self.position.z + random.uniform(-self.size.z / 2, self.size.z / 2)
            )
            self.neurons.append(Neuron(self, pos))

    def setup_interfaces(self):
        """Designate neurons on the faces of the cuboid as interface neurons."""
        # Simple interface setup: neurons closest to the -Z face are input, +Z are output
        # A more robust implementation would use a grid
        all_neurons_sorted_z = sorted(self.neurons, key=lambda n: n.position.z)
        
        if self.is_input:
            # Let's define the input interface as a 10x10 grid on the -Z face
            self.input_interface = self._find_closest_neurons_to_plane('z', self.position.z - self.size.z / 2, 100)

        if self.is_output:
            # Output interface on the +Z face
            self.output_interface = self._find_closest_neurons_to_plane('z', self.position.z + self.size.z / 2, 10) # 10 output neurons for 10 digits

    def _find_closest_neurons_to_plane(self, axis, value, count):
        """Helper to find 'count' neurons closest to a plane."""
        axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
        
        def get_pos(neuron):
            return neuron.position.to_tuple()[axis_idx]

        sorted_neurons = sorted(self.neurons, key=lambda n: abs(get_pos(n) - value))
        return sorted_neurons[:count]


    async def run_cortex(self):
        """Start all neuron tasks and manage cortex-level processes."""
        neuron_tasks = [asyncio.create_task(n.run()) for n in self.neurons]
        
        # Cortex-level tasks
        async def manage_energy():
            while True:
                self.energy_level += CORTEX_ENERGY_REGEN_RATE
                if self.energy_level > CORTEX_ENERGY_CAPACITY:
                    self.energy_level = CORTEX_ENERGY_CAPACITY
                await asyncio.sleep(1)
        
        async def manage_connections():
            while True:
                # Periodically try to connect unconnected neurons
                await self.establish_internal_connections()
                await asyncio.sleep(5)

        energy_task = asyncio.create_task(manage_energy())
        connection_task = asyncio.create_task(manage_connections())
        
        await asyncio.gather(*neuron_tasks, energy_task, connection_task)

    def request_energy(self, pos: Vector3, amount: float) -> float:
        """A neuron requests energy. Cortex provides it if available."""
        # For simplicity, energy is uniformly available. A more complex model
        # could have energy gradients.
        if self.energy_level >= amount:
            self.energy_level -= amount
            return amount
        else:
            available = self.energy_level
            self.energy_level = 0
            return available

    async def establish_internal_connections(self):
        """Attempt to form connections between neurons within this cortex."""
        unconnected_axons = [n.axon for n in self.neurons if not n.axon.is_connected() and not n.is_stagnant]
        unconnected_dendrites = [d for n in self.neurons for d in n.dendrites if not d.is_connected() and not n.is_stagnant]
        
        random.shuffle(unconnected_axons)
        random.shuffle(unconnected_dendrites)

        for axon in unconnected_axons:
            for dendrite in unconnected_dendrites:
                if axon.neuron != dendrite.neuron:
                    dist = axon.neuron.position.distance_to(dendrite.neuron.position)
                    if dist < NEURON_CONNECTION_RADIUS:
                        if not dendrite.is_connected():
                            await axon.connect(dendrite)
                            unconnected_dendrites.remove(dendrite)
                            break # Axon is now connected

    async def find_connection_for(self, searching_neuron: Neuron):
        """Find a connection for a neuron whose growth sphere is active."""
        if not searching_neuron.growth_sphere_active:
            return

        sphere_pos = searching_neuron.position
        sphere_rad = searching_neuron.growth_sphere_radius

        # Find potential dendrites to connect to
        potential_dendrites = [
            d for n in self.neurons 
            for d in n.dendrites 
            if not d.is_connected() and n.id != searching_neuron.id and not n.is_stagnant
        ]

        for dendrite in potential_dendrites:
            dist = sphere_pos.distance_to(dendrite.neuron.position)
            if dist < sphere_rad:
                print(f"Growth sphere connection made from {searching_neuron.id} to {dendrite.neuron.id}")
                await searching_neuron.axon.connect(dendrite)
                break # Stop after first connection

    def reward(self):
        """Give a burst of energy to the cortex for a correct answer."""
        self.energy_level += CORTEX_REWARD_ENERGY
        print("Cortex Rewarded!")

class VirtualEnvironment:
    """Manages cortices, data, and visualization."""
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Neural Cortex Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 16)
        self.camera = Camera(pos=Vector3(150, 150, -400))
        self.cortexes = []
        self.running = True
        self.mouse_dragging = False
        self.mnist_data = self.load_mnist_like_data()
        self.current_stimulus = None
        self.current_label = None
        self.correct_predictions = 0
        self.total_predictions = 0

    def load_mnist_like_data(self):
        """Generates dummy data in the MNIST format (28x28 images)."""
        print("Loading dummy MNIST data...")
        data = []
        for i in range(10): # 10 classes (digits 0-9)
            for _ in range(50): # 50 examples per class
                img = np.zeros((28, 28), dtype=float)
                # Create a simple pattern for each digit
                if i == 0: # Circle
                    for r in range(10, 14):
                        for angle in range(360):
                            x = int(14 + r * math.cos(math.radians(angle)))
                            y = int(14 + r * math.sin(math.radians(angle)))
                            if 0 <= x < 28 and 0 <= y < 28: img[y, x] = 1.0
                elif i == 1: # Vertical line
                    img[5:23, 13:15] = 1.0
                elif i == 2:
                    img[5:7, 8:20] = 1.0; img[13:15, 8:20] = 1.0; img[21:23, 8:20] = 1.0
                    img[7:14, 18:20] = 1.0; img[15:22, 8:10] = 1.0
                # ... add more patterns for other digits
                else: # Default: diagonal line
                    for j in range(5, 23): img[j,j] = 1.0

                data.append((img, i))
        random.shuffle(data)
        return data

    def add_cortex(self, cortex: Cortex):
        self.cortexes.append(cortex)

    def project(self, pos3d: Vector3):
        """Projects a 3D point to 2D screen coordinates."""
        # A very basic perspective projection
        view_mat = self.camera.get_view_matrix()
        pos4d = np.array([pos3d.x, pos3d.y, pos3d.z, 1])
        
        transformed = np.dot(pos4d, view_mat)
        
        # This is a simplified perspective division
        z = transformed[2]
        if z >= 0: # Behind camera
            return None
            
        fov = 250
        x = transformed[0] * fov / z + SCREEN_WIDTH / 2
        y = -transformed[1] * fov / z + SCREEN_HEIGHT / 2
        
        return int(x), int(y)

    def draw_text(self, text, pos, color=FONT_COLOR):
        label = self.font.render(text, 1, color)
        self.screen.blit(label, pos)

    def draw(self):
        """Main drawing loop."""
        self.screen.fill(BACKGROUND_COLOR)

        for cortex in self.cortexes:
            # Draw cortex bounds
            c_pos = cortex.position
            c_size = cortex.size
            p = [
                c_pos + Vector3(-c_size.x/2, -c_size.y/2, -c_size.z/2),
                c_pos + Vector3(c_size.x/2, -c_size.y/2, -c_size.z/2),
                c_pos + Vector3(c_size.x/2, c_size.y/2, -c_size.z/2),
                c_pos + Vector3(-c_size.x/2, c_size.y/2, -c_size.z/2),
                c_pos + Vector3(-c_size.x/2, -c_size.y/2, c_size.z/2),
                c_pos + Vector3(c_size.x/2, -c_size.y/2, c_size.z/2),
                c_pos + Vector3(c_size.x/2, c_size.y/2, c_size.z/2),
                c_pos + Vector3(-c_size.x/2, c_size.y/2, c_size.z/2),
            ]
            points = [self.project(v) for v in p]
            
            # Draw neurons and connections
            for neuron in cortex.neurons:
                pos2d = self.project(neuron.position)
                if pos2d:
                    # Determine neuron color based on state
                    if neuron.is_stagnant:
                        color = (50, 50, 70) # Dark blue/grey for stagnant
                        radius = 1
                    else:
                        # Color by energy level (blue to yellow)
                        energy_ratio = min(neuron.energy / 50.0, 1.0)
                        color = (int(50 + 200 * energy_ratio), int(200 * energy_ratio), int(255 - 200 * energy_ratio))
                        radius = 2 + int(neuron.puffer / NEURON_PUFFER_CAPACITY * 3)
                    pygame.draw.circle(self.screen, color, pos2d, radius)

                    # Draw growth sphere
                    if neuron.growth_sphere_active:
                        # This is a 2D representation of the 3D sphere
                        p1 = self.project(neuron.position + Vector3(neuron.growth_sphere_radius, 0, 0))
                        if p1:
                            screen_radius = int(abs(p1[0] - pos2d[0]))
                            if screen_radius > 0:
                                s = pygame.Surface((screen_radius*2, screen_radius*2), pygame.SRCALPHA)
                                pygame.draw.circle(s, SPHERE_COLOR, (screen_radius, screen_radius), screen_radius)
                                self.screen.blit(s, (pos2d[0] - screen_radius, pos2d[1] - screen_radius))

                    # Draw axon connections
                    if neuron.axon.is_connected():
                        start_pos = pos2d
                        end_pos = self.project(neuron.axon.connection.neuron.position)
                        if end_pos:
                            # Make connection glow if recently fired
                            if time.time() - neuron.axon.last_emit_time < 0.1:
                                pygame.draw.line(self.screen, ACTIVE_CONNECTION_COLOR, start_pos, end_pos, 2)
                            else:
                                pygame.draw.line(self.screen, CONNECTION_COLOR, start_pos, end_pos, 1)
        
        # Draw UI
        self.draw_hud()
        pygame.display.flip()

    def draw_hud(self):
        """Draws heads-up display information."""
        self.draw_text(f"FPS: {self.clock.get_fps():.1f}", (10, 10))
        self.draw_text(f"Camera Pos: ({self.camera.pos.x:.0f}, {self.camera.pos.y:.0f}, {self.camera.pos.z:.0f})", (10, 30))
        self.draw_text("Controls: WASD + Shift/Space + Mouse Drag", (10, 50))
        
        for i, cortex in enumerate(self.cortexes):
            energy_percent = (cortex.energy_level / CORTEX_ENERGY_CAPACITY) * 100
            self.draw_text(f"Cortex {i} Energy: {energy_percent:.1f}%", (10, 90 + i * 20))

        # Draw MNIST stimulus
        if self.current_stimulus is not None:
            self.draw_text("Current Input:", (SCREEN_WIDTH - 180, 10))
            self.draw_text(f"True Label: {self.current_label}", (SCREEN_WIDTH - 180, 30))
            stimulus_surface = pygame.Surface((28*5, 28*5))
            stimulus_surface.fill((40,40,50))
            for y, row in enumerate(self.current_stimulus):
                for x, val in enumerate(row):
                    if val > 0:
                        pygame.draw.rect(stimulus_surface, (255, 255, 255), (x*5, y*5, 5, 5))
            self.screen.blit(stimulus_surface, (SCREEN_WIDTH - 180, 50))

        # Draw accuracy
        accuracy = (self.correct_predictions / self.total_predictions * 100) if self.total_predictions > 0 else 0
        self.draw_text(f"Accuracy: {accuracy:.2f}% ({self.correct_predictions}/{self.total_predictions})", (SCREEN_WIDTH - 250, SCREEN_HEIGHT - 30))


    def handle_events(self):
        """Handle user input."""
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.mouse_dragging = True
                    pygame.event.set_grab(True)
                    pygame.mouse.set_visible(False)
            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:
                    self.mouse_dragging = False
                    pygame.event.set_grab(False)
                    pygame.mouse.set_visible(True)
        
        keys = pygame.key.get_pressed()
        mouse_rel = (0,0)
        if self.mouse_dragging:
            mouse_rel = pygame.mouse.get_rel()

        self.camera.process_input(keys, mouse_rel)


    async def stimulation_loop(self):
        """The main loop for presenting data and checking results."""
        input_cortex = self.cortexes[0]
        output_cortex = self.cortexes[-1]

        for img, label in self.mnist_data:
            if not self.running: break

            self.current_stimulus = img
            self.current_label = label
            self.total_predictions += 1
            
            # Present the image to the input interface
            # Map 28x28 image to the 10x10 input interface (100 neurons)
            if len(input_cortex.input_interface) == 100:
                for i in range(100):
                    # Simple downsampling
                    img_x = int((i % 10) * 2.8)
                    img_y = int((i // 10) * 2.8)
                    pixel_value = img[img_y, img_x]
                    
                    if pixel_value > 0.5:
                        neuron = input_cortex.input_interface[i]
                        # Directly inject charge into the input neuron's puffer
                        charge = pixel_value * NEURON_PUFFER_CAPACITY
                        await neuron.receive_charge(charge, None) # Source is external
            
            # Wait for the network to process
            await asyncio.sleep(2.0)

            # Read the output
            output_activity = [n.puffer for n in output_cortex.output_interface]
            if not any(output_activity):
                prediction = -1 # No activity
            else:
                prediction = np.argmax(output_activity)

            print(f"Presented: {label}, Network Guessed: {prediction}, Activity: {[f'{a:.1f}' for a in output_activity]}")

            # Check accuracy and provide feedback
            if prediction == label:
                self.correct_predictions += 1
                for cortex in self.cortexes:
                    cortex.reward()
            
            # Wait before next image
            await asyncio.sleep(3.0)
        
        print("Finished dataset.")


    async def main_loop(self):
        """The main async loop combining simulation and visualization."""
        cortex_tasks = [asyncio.create_task(c.run_cortex()) for c in self.cortexes]
        stim_task = asyncio.create_task(self.stimulation_loop())

        while self.running:
            self.handle_events()
            self.draw()
            self.clock.tick(FPS)
            await asyncio.sleep(0) # Yield control to the event loop

        # Cleanup
        for task in cortex_tasks + [stim_task]:
            task.cancel()
        pygame.quit()


async def main():
    env = VirtualEnvironment()

    # Create two cortices and link them
    cortex1 = Cortex(position=Vector3(0, 0, 0), size=Vector3(200, 200, 50), neuron_count=200, is_input=True)
    cortex2 = Cortex(position=Vector3(0, 0, 150), size=Vector3(100, 100, 30), neuron_count=50, is_output=True)

    env.add_cortex(cortex1)
    env.add_cortex(cortex2)
    
    # Simple inter-cortex connection logic
    # Connect some output neurons of cortex1 to input neurons of cortex2
    # This is a placeholder for a more sophisticated connection algorithm
    print("Connecting cortices...")
    for axon_neuron in cortex1._find_closest_neurons_to_plane('z', cortex1.position.z + cortex1.size.z/2, 20):
        for dendrite_neuron in cortex2._find_closest_neurons_to_plane('z', cortex2.position.z - cortex2.size.z/2, 20):
            # Find a free dendrite on the target neuron
            free_dendrite = next((d for d in dendrite_neuron.dendrites if not d.is_connected()), None)
            if free_dendrite:
                await axon_neuron.axon.connect(free_dendrite)
                break # Move to next axon

    await env.main_loop()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Simulation stopped by user.")