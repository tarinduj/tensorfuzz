import gzip
import os
import shutil
import tempfile

import jax.numpy as jnp
import numpy as np
from six.moves import urllib

def read32(bytestream):
    """Read 4 bytes from bytestream as an unsigned 32-bit integer."""
    dtype = np.dtype(np.uint32).newbyteorder(">")
    return np.frombuffer(bytestream.read(4), dtype=dtype)[0].astype(np.uint32)

def check_image_file_header(filename):
    """Validate that filename corresponds to images for the MNIST dataset."""
    with open(filename, "rb") as fptr:
        magic = read32(fptr)
        read32(fptr)  # num_images, unused
        rows = read32(fptr)
        cols = read32(fptr)
        if magic != 2051:
            raise ValueError(
                "Invalid magic number %d in MNIST file %s" % (magic, fptr.name)
            )
        if rows != 28 or cols != 28:
            raise ValueError(
                "Invalid MNIST file %s: Expected 28x28 images, found %dx%d"
                % (fptr.name, rows, cols)
            )

def check_labels_file_header(filename):
    """Validate that filename corresponds to labels for the MNIST dataset."""
    with open(filename, "rb") as fptr:
        magic = read32(fptr)
        read32(fptr)  # num_items, unused
        if magic != 2049:
            raise ValueError(
                "Invalid magic number %d in MNIST file %s" % (magic, fptr.name)
            )

def download(directory, filename):
    """Download (and unzip) a file from the MNIST dataset if not already done."""
    filepath = os.path.join(directory, filename)
    if os.path.exists(filepath):
        return filepath
    if not os.path.exists(directory):
        os.makedirs(directory)
    url = "https://storage.googleapis.com/cvdf-datasets/mnist/" + filename + ".gz"
    _, zipped_filepath = tempfile.mkstemp(suffix=".gz")
    print("Downloading %s to %s" % (url, zipped_filepath))
    urllib.request.urlretrieve(url, zipped_filepath)
    with gzip.open(zipped_filepath, "rb") as f_in, open(filepath, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(zipped_filepath)
    return filepath

def dataset(directory, images_file, labels_file):
    """Download and parse MNIST dataset."""

    images_file = download(directory, images_file)
    labels_file = download(directory, labels_file)

    check_image_file_header(images_file)
    check_labels_file_header(labels_file)

    def decode_image(image):
        """Decodes and normalizes the image."""
        image = jnp.frombuffer(image, dtype=jnp.uint8)
        image = image.astype(jnp.float32)
        image = image.reshape([784])
        image = image / 255.0  # [0.0, 1.0]
        image = image * 2.0  # [0.0, 2.0]
        image = image - 1.0  # [-1.0, 1.0]
        return image

    def decode_label(label):
        """Decodes and reshapes the label."""
        label = jnp.frombuffer(label, dtype=jnp.uint8)
        return label.astype(jnp.int32)

    images_data = np.frombuffer(open(images_file, 'rb').read()[16:], dtype=np.uint8)  # skip header
    labels_data = np.frombuffer(open(labels_file, 'rb').read()[8:], dtype=np.uint8)  # skip header

    images = jnp.array([decode_image(image) for image in np.split(images_data, len(images_data) // 784)])
    labels = jnp.array([decode_label(label) for label in labels_data])

    return images, labels

def train(directory):
    """Dataset object for MNIST training data."""
    return dataset(directory, "train-images-idx3-ubyte", "train-labels-idx1-ubyte")

def test(directory):
    """Dataset object for MNIST test data."""
    return dataset(directory, "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")
