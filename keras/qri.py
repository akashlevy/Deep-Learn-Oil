"""Library for QRI data to work with Keras"""

import cPickle, gzip


def load_data(filename):
    """Load datasets from a file"""
    with gzip.open(filename, "rb") as file:
        return cPickle.load(file)