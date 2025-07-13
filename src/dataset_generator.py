import numpy as np
import random
import string
from tqdm import tqdm

def generate_random_feature():
    """Generates a random feature name."""
    return ''.join(random.choices(string.ascii_lowercase, k=4))

def create_object_template(num_features):
    """Creates a template for a single 3D object class with unique feature locations."""
    features = {}
    max_attempts = num_features * 100
    attempts = 0

    while len(features) < num_features and attempts < max_attempts:
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.arccos(2 * np.random.uniform(0, 1) - 1)

        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)

        location = (round(x, 2), round(y, 2), round(z, 2))

        if location not in features:
            features[location] = generate_random_feature()

        attempts += 1

    if len(features) < num_features:
        print(f"Warning: Could only generate {len(features)} unique features out of {num_features} requested.")

    return features

def generate_dataset(num_object_types, features_per_object, dataset_size, desc=""):
    """Generates a full dataset of object templates with a progress bar."""
    dataset = {}
    for i in tqdm(range(num_object_types), desc=f"Generating {desc} dataset"):
        obj_name = f"Object-{i+1}"
        instances = [create_object_template(features_per_object) for _ in range(dataset_size)]
        dataset[obj_name] = instances
    return dataset