
"""Library for character data to work with Keras"""

def load_data(filename):
    """Load datasets from a file"""
    with gzip.open(filename, "rb") as file:
        return cPickle.load(file)