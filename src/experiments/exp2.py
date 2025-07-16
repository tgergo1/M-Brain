import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import gzip
import os
import logging
import sys
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)

class Neuron:
    def __init__(self, x, y, z, tau=20.0, v_rest=-70.0):
        self.x, self.y, self.z = x, y, z
        self.v = v_rest
        self.u = -14.0
        self.tau = tau
        self.v_rest = v_rest
        self.spike = False
        self.connections = []
        self.energy = 0.0

    def step(self, dt=1.0, I=0.0):
        a, b, c, d = 0.02, 0.2, -65.0, 8.0
        if self.spike:
            self.v = c
            self.u += d
            self.spike = False
        self.v += dt * (0.04 * self.v**2 + 5 * self.v + 140 - self.u + I)
        self.u += dt * a * (b * self.v - self.u)
        if self.v >= 30:
            self.spike = True
            self.v = 30
        return self.spike

class Synapse:
    def __init__(self, pre, post, w=1.0, delay=1):
        self.pre, self.post = pre, post
        self.w = w
        self.delay = delay
        self.trace = 0.0
        self.energy_gradient = 0.0

class Layer:
    def __init__(self, name, size, span):
        self.name = name
        self.neurons = []
        self.span = span
        for i in range(size):
            x = np.random.uniform(*span[0])
            y = np.random.uniform(*span[1])
            z = np.random.uniform(*span[2])
            self.neurons.append(Neuron(x, y, z))

class EnergyModel:
    def __init__(self, layers, lr=0.001):
        self.layers = layers
        self.lr = lr
        self.synapses = []
        self._build_connections()

    def _build_connections(self):
        logging.info("Building synapses between layers...")
        total = 0
        for i in range(len(self.layers) - 1):
            l1, l2 = self.layers[i], self.layers[i + 1]
            for n1 in l1.neurons:
                for n2 in l2.neurons:
                    w = np.random.normal(0.5, 0.1)
                    syn = Synapse(n1, n2, w)
                    self.synapses.append(syn)
                    n1.connections.append(syn)
                    total += 1
        logging.info(f"Created {total} synapses")

    def feed(self, img):
        img = img.flatten()
        for idx, val in enumerate(img):
            self.layers[0].neurons[idx].v += val * 50

    def free_phase(self, duration=50):
        for _ in range(duration):
            for syn in self.synapses:
                if syn.pre.spike:
                    syn.trace += 1.0
                    I = syn.w * syn.trace
                    syn.post.v += I
                    syn.post.energy += abs(I)
                syn.trace *= 0.95
            for layer in self.layers:
                for n in layer.neurons:
                    n.step()
                    n.energy += n.v**2

    def clamp_phase(self, label, duration=50):
        target = np.zeros(10)
        target[label] = 1.0
        out_layer = self.layers[-1]
        for _ in range(duration):
            for i, n in enumerate(out_layer.neurons):
                n.v = target[i] * 30
            self.free_phase(duration=1)

    def update_weights(self):
        for syn in self.synapses:
            syn.energy_gradient = syn.post.energy - syn.pre.energy
            syn.w -= self.lr * syn.energy_gradient
            syn.w = np.clip(syn.w, 0, 2)

    def energy(self):
        return sum(n.energy for layer in self.layers for n in layer.neurons)

class ThousandsBrain:
    def __init__(self):
        logging.info("Initializing ThousandsBrain…")
        self.layers = [
            Layer("input", 784, [(0, 28), (0, 28), (0, 1)]),
            Layer("hidden1", 400, [(0, 20), (0, 20), (1, 5)]),
            Layer("hidden2", 200, [(0, 15), (0, 15), (5, 10)]),
            Layer("output", 10, [(0, 5), (0, 5), (10, 12)])
        ]
        self.model = EnergyModel(self.layers)

    def predict(self, img):
        for layer in self.model.layers:
            for n in layer.neurons:
                n.v = -70
                n.energy = 0
                n.spike = False
        self.model.feed(img)
        self.model.free_phase(duration=30)
        out = [n.v for n in self.model.layers[-1].neurons]
        return np.argmax(out)

    def train_step(self, img, label):
        self.predict(img)
        self.model.clamp_phase(label, duration=20)
        self.model.update_weights()

def load_mnist():
    path = 'mnist.pkl.gz'
    if not os.path.exists(path):
        logging.info("MNIST file not found locally; downloading…")
        import urllib.request
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/mnielsen/neural-networks-and-deep-learning/master/data/mnist.pkl.gz",
            path
        )
        logging.info("Download complete.")
    logging.info("Loading MNIST…")
    with gzip.open(path, 'rb') as f:
        return pickle.load(f, encoding='latin1')

def save_model(model, path='brain.pkl'):
    logging.info(f"Saving model to {path}")
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def load_model(path='brain.pkl'):
    logging.info(f"Loading model from {path}")
    with open(path, 'rb') as f:
        return pickle.load(f)

def evaluate(brain, x_test, y_test):
    logging.info("Evaluating model…")
    preds = []
    for i, x in enumerate(tqdm(x_test, desc="Evaluation")):
        preds.append(brain.predict(x.reshape(28, 28)))
    acc = np.mean(np.array(preds) == y_test)
    logging.info(f"Evaluation accuracy: {acc:.4f} ({int(acc*len(y_test))}/{len(y_test)} correct)")
    return acc

def visualize_3d(brain):
    logging.info("Visualizing 3D neuron positions…")
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['red', 'green', 'blue', 'orange']
    for i, layer in enumerate(brain.layers):
        xs = [n.x for n in layer.neurons]
        ys = [n.y for n in layer.neurons]
        zs = [n.z for n in layer.neurons]
        ax.scatter(xs, ys, zs, c=colors[i % len(colors)], label=layer.name)
    ax.legend()
    plt.title("M-Brain")
    plt.show()

if __name__ == "__main__":
    logging.info("Starting M-Brain experiment")
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_mnist()
    logging.info(f"MNIST loaded: {len(x_train)} train, {len(x_valid)} valid, {len(x_test)} test")
    brain = ThousandsBrain()
    visualize_3d(brain)

    train_epochs = 3
    batch = 1000
    for epoch in range(train_epochs):
        logging.info(f"Epoch {epoch + 1}/{train_epochs}")
        idx = np.random.choice(len(x_train), batch, replace=False)
        for i in tqdm(idx, desc="Training"):
            brain.train_step(x_train[i], y_train[i])
        acc = evaluate(brain, x_test[:1000], y_test[:1000])
        logging.info(f"Epoch {epoch + 1} accuracy: {acc:.4f}")
        save_model(brain)
    final_acc = evaluate(brain, x_test, y_test)
    logging.info(f"Final test accuracy: {final_acc:.4f}")