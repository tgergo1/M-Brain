import numpy as np
import random
import string

def generate_random_feature():
    """Generates a random feature name."""
    return ''.join(random.choices(string.ascii_lowercase, k=4))

def create_object_template(num_features):
    """Creates a template for a single 3D object class."""
    features = {}
    # Generate features on the surface of a unit sphere for simplicity
    for _ in range(num_features):
        # Use spherical coordinates to ensure points are on a surface
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.arccos(2 * np.random.uniform(0, 1) - 1)
        
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        
        # Round to create a discrete grid-like structure on the sphere
        location = (round(x, 2), round(y, 2), round(z, 2))
        feature = generate_random_feature()
        features[location] = feature
    return features

def generate_dataset(num_object_types, features_per_object, dataset_size):
    """Generates a full dataset of object templates."""
    dataset = {}
    for i in range(num_object_types):
        obj_name = f"Object-{i+1}"
        # Generate multiple instances for each object type
        instances = [create_object_template(features_per_object) for _ in range(dataset_size)]
        dataset[obj_name] = instances
    return dataset