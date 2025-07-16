#!/usr/bin/env python3
"""
Thousands-Brain-inspired Energy–based 3-D Neuronal Model on MNIST
Fixed incentive structure:
1. Stronger, location-coded input → mini-column routing
2. Sparse, competitive inhibition inside each mini-column
3. Reward-like global scalar feedback (energy = –correct + log Σexp(others))
4. Hebbian + homeostatic plasticity
"""

import os, time, json, gzip, pickle, struct, logging, hashlib, urllib.request
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numba import njit, prange
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

MNIST_URL = {
    "train_images": "https://github.com/fgnt/mnist/raw/master/train-images-idx3-ubyte.gz",
    "train_labels": "https://github.com/fgnt/mnist/raw/master/train-labels-idx1-ubyte.gz",
    "test_images":  "https://github.com/fgnt/mnist/raw/master/t10k-images-idx3-ubyte.gz",
    "test_labels":  "https://github.com/fgnt/mnist/raw/master/t10k-labels-idx1-ubyte.gz"
}
CACHE = "mnist_cache.pkl"
MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

###############################################################################
# Parameters
###############################################################################
SCALE = 7                               # 7³ = 343 mini-columns
NEURON_GRID = (SCALE, SCALE, SCALE)
NEURONS_PER_COL = 10
TOTAL_NEURONS = np.prod(NEURON_GRID) * NEURONS_PER_COL
INPUT_SIZE = 28 * 28
OUTPUT_CLASSES = 10
CONNECTIVITY = 0.05
DT = 1.0
TAU_MEM = 20.0
TAU_SYN = 5.0
TAU_PLAST = 1000.0
V_REST = -70.0
V_TH = -55.0
R = 1.0
ETA = 5e-3
SPARSITY_TARGET = 0.05
np.random.seed(42)

###############################################################################
# Data
###############################################################################
def download(url, path):
    if not os.path.isfile(path):
        logging.info(f"Downloading {url}")
        urllib.request.urlretrieve(url, path)
def load_mnist():
    if os.path.isfile(CACHE):
        return pickle.load(open(CACHE, "rb"))
    paths = {k: f"{k}.gz" for k in MNIST_URL}
    for k, u in MNIST_URL.items(): download(u, paths[k])
    def _read_int(f): return struct.unpack(">I", f.read(4))[0]
    def _load_images(p):
        with gzip.open(p, "rb") as f:
            _, n, r, c = struct.unpack(">IIII", f.read(16))
            return (np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r*c)/255.0).astype(np.float32)
    def _load_labels(p):
        with gzip.open(p, "rb") as f:
            _, _ = struct.unpack(">II", f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)
    data = {k: _load_images(paths[k]) if "images" in k else _load_labels(paths[k]) for k in MNIST_URL}
    pickle.dump(data, open(CACHE, "wb"))
    logging.info("MNIST loaded")
    return data

###############################################################################
# 3-D Tissue
###############################################################################
class BrainTissue:
    def __init__(self):
        self.neurons = []
        idx = 0
        for i in range(NEURON_GRID[0]):
            for j in range(NEURON_GRID[1]):
                for k in range(NEURON_GRID[2]):
                    for _ in range(NEURONS_PER_COL):
                        self.neurons.append((i, j, k, idx))
                        idx += 1
        self.n = len(self.neurons)
        self.pos = np.array([n[:3] for n in self.neurons], dtype=np.float32)
        self.V = np.full(self.n, V_REST, dtype=np.float32)
        self.I = np.zeros(self.n, dtype=np.float32)
        self.spike_log = np.zeros(self.n, dtype=np.float32)
        self.W_in = self._build_input_weights()
        self.W_rec = self._sparse_recurrent()
        self.W_out = np.random.normal(0, 0.1, (self.n, OUTPUT_CLASSES)).astype(np.float32)
        self.inhib = self._inhibition_mask()
    def _build_input_weights(self):
        # Each mini-column gets a localized receptive field on the 28×28 image
        W = np.zeros((INPUT_SIZE, self.n), dtype=np.float32)
        col_size = 28 // NEURON_GRID[0]
        for col_idx, (cx, cy, cz, _) in enumerate(self.neurons):
            x0, y0 = cx * col_size, cy * col_size
            mask = np.zeros((28, 28))
            mask[y0:y0+col_size, x0:x0+col_size] = 1.0
            mask = mask.flatten()
            W[:, col_idx] = mask * np.random.normal(0.5, 0.1, INPUT_SIZE)
        return W
    def _sparse_recurrent(self):
        total = self.n * self.n
        nnz = int(CONNECTIVITY * total)
        rows = np.random.randint(0, self.n, nnz)
        cols = np.random.randint(0, self.n, nnz)
        data = np.random.normal(0, 0.05, nnz).astype(np.float32)
        W = np.zeros((self.n, self.n), dtype=np.float32)
        W[rows, cols] = data
        np.fill_diagonal(W, 0)
        return W
    def _inhibition_mask(self):
        # Strong lateral inhibition inside each mini-column
        mask = np.zeros((self.n, self.n), dtype=np.float32)
        col_size = NEURONS_PER_COL
        for c in range(np.prod(NEURON_GRID)):
            start = c * col_size
            end = start + col_size
            for i in range(start, end):
                mask[i, start:end] = -2.0
                mask[i, i] = 0.0
        return mask
    def reset(self):
        self.V[:] = V_REST
        self.I[:] = 0.0
        self.spike_log[:] = 0.0

###############################################################################
# Numba simulation
###############################################################################
@njit(parallel=True, fastmath=True)
def step_network(V, I, W_rec, W_in, img, spike_log, inhib, dt, tau_mem, tau_syn, v_rest, v_th, r):
    n = V.shape[0]
    spike = np.zeros(n, dtype=np.bool_)
    for i in prange(n):
        dv = (-(V[i] - v_rest) + r * I[i]) * dt / tau_mem
        V[i] += dv
        if V[i] >= v_th:
            spike[i] = True
            V[i] = v_rest
    I *= np.exp(-dt / tau_syn)
    I *= 0.9
    spikes = spike.astype(np.float32)
    I += spikes @ W_rec
    I += img @ W_in
    I += spikes @ inhib
    spike_log += spikes
    return spikes

###############################################################################
# Training & energy
###############################################################################
class Trainer:
    def __init__(self, tissue):
        self.tissue = tissue
    def energy(self, spike_log, label):
        out = spike_log @ self.tissue.W_out
        correct = out[label]
        max_other = np.max(np.delete(out, label))
        return -correct + max_other
    def update(self, img, spike_log, label, energy):
        # Output layer
        out = spike_log @ self.tissue.W_out
        softmax = np.exp(out - np.max(out))
        softmax /= softmax.sum()
        softmax[label] -= 1
        dW_out = np.outer(spike_log, softmax)
        self.tissue.W_out -= ETA * dW_out
        # Homeostatic sparsity on recurrent
        avg_rate = spike_log.mean()
        homeo = (SPARSITY_TARGET - avg_rate) * ETA * 0.1
        self.tissue.W_rec += homeo * self.tissue.W_rec
        self.tissue.W_rec = np.clip(self.tissue.W_rec, -0.5, 0.5)
        np.fill_diagonal(self.tissue.W_rec, 0)

###############################################################################
# Epoch
###############################################################################
def run_epoch(images, labels, tissue, trainer, log_every=500):
    idx = np.arange(len(images))
    np.random.shuffle(idx)
    correct = 0
    for i, (img, label) in enumerate(zip(images[idx], labels[idx])):
        tissue.reset()
        spike_log = np.zeros(tissue.n, dtype=np.float32)
        for _ in range(30):
            step_network(tissue.V, tissue.I, tissue.W_rec, tissue.W_in, img,
                         spike_log, tissue.inhib, DT, TAU_MEM, TAU_SYN, V_REST, V_TH, R)
        pred = np.argmax(spike_log @ tissue.W_out)
        if pred == label: correct += 1
        energy = trainer.energy(spike_log, label)
        trainer.update(img, spike_log, label, energy)
        if i % log_every == 0:
            logging.info(f"[{i:5d}/{len(images)}] acc={correct/(i+1):.3f} e={energy:.3f}")
    return correct / len(images)

###############################################################################
# Evaluate
###############################################################################
def evaluate(images, labels, tissue):
    correct = 0
    for img, label in zip(images, labels):
        tissue.reset()
        spike_log = np.zeros(tissue.n, dtype=np.float32)
        for _ in range(30):
            step_network(tissue.V, tissue.I, tissue.W_rec, tissue.W_in, img,
                         spike_log, tissue.inhib, DT, TAU_MEM, TAU_SYN, V_REST, V_TH, R)
        pred = np.argmax(spike_log @ tissue.W_out)
        if pred == label: correct += 1
    acc = correct / len(images)
    logging.info(f"Test accuracy = {acc:.4f}")
    return acc

###############################################################################
# Save / Load
###############################################################################
def save_model(tissue, epoch, acc):
    fname = f"{MODEL_DIR}/brain_e{epoch}_acc{acc:.3f}.npz"
    np.savez_compressed(fname, W_rec=tissue.W_rec, W_in=tissue.W_in,
                        W_out=tissue.W_out, pos=tissue.pos)
    logging.info(f"Saved {fname}")
def load_model(fname, tissue):
    data = np.load(fname)
    tissue.W_rec[:] = data["W_rec"]
    tissue.W_in[:] = data["W_in"]
    tissue.W_out[:] = data["W_out"]

###############################################################################
# Main
###############################################################################
def main():
    data = load_mnist()
    tissue = BrainTissue()
    trainer = Trainer(tissue)
    for ep in range(1, 4):
        logging.info(f"=== Epoch {ep} ===")
        acc = run_epoch(data["train_images"], data["train_labels"], tissue, trainer)
        test_acc = evaluate(data["test_images"], data["test_labels"], tissue)
        save_model(tissue, ep, test_acc)

if __name__ == "__main__":
    main()