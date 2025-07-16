import tensorflow as tf
import tensorflow_datasets as tfds
import jax.numpy as jnp
from tqdm import tqdm

def load_mnist_data(train_limit=60000, test_limit=10000, padding=0):
    """
    Loads, normalizes, and pads the MNIST dataset using TensorFlow Datasets.
    
    Args:
        train_limit (int): Max number of training samples to load.
        test_limit (int): Max number of test samples to load.
        padding (int): Amount of zero-padding to add around each image.

    Returns:
        Tuple of JAX arrays: (train_images, train_labels, test_images, test_labels)
    """
    # Disable TensorFlow's GPU memory allocation to leave VRAM for JAX
    tf.config.set_visible_devices([], 'GPU')

    # Load the dataset
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    def normalize_and_pad(image, label):
        """Converts image to float, normalizes, and adds padding."""
        image = tf.cast(image, tf.float32) / 255.0
        if padding > 0:
            paddings = [[padding, padding], [padding, padding], [0, 0]]
            image = tf.pad(image, paddings)
        return image, label

    # Process the datasets
    ds_train = ds_train.map(normalize_and_pad, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.take(train_limit).cache().batch(train_limit)
    
    ds_test = ds_test.map(normalize_and_pad, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.take(test_limit).cache().batch(test_limit)

    # Extract numpy arrays and convert to JAX arrays
    train_images, train_labels = next(iter(tfds.as_numpy(ds_train)))
    test_images, test_labels = next(iter(tfds.as_numpy(ds_test)))

    # Squeeze the channel dimension as it's grayscale
    return jnp.squeeze(train_images), jnp.array(train_labels), jnp.squeeze(test_images), jnp.array(test_labels)