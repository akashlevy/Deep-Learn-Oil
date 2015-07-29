import cPickle
import gzip
import os

# Shape of data
N_SEQ = 
N_STEPS = 3
N_IN = 90
N_OUT = 10

# Proportion of text dedicated to dataset
P_TRAIN = 0.8
P_VALID = 0.1
P_TEST = 0.1

def load_data(dataset="shakespeare", max_size=250000):
    """
    Loads a PlainText file into a string without '\n'.
    Returns the string and the length of the string.

    :type dataset: String
    :param dataset: path to dataset
    :type max_size: int
    :param max_size: max number of characters
    :type valid_portion: float
    :param valid_portion: proportion of full train set used for validation set
    """
    #############
    # LOAD TEXT #
    #############
    path = dataset + ".pkl.gz"

    if not os.path.exists(path):
        path = dataset + ".txt"
        train_set, valid_set, test_set, unique_characters = process_data(path, max_size)
    else:
        f = gzip.open(path, "rb")
        train_set, valid_set, test_set, unique_characters = cPickle.load(f)
        f.close()

    return train_set, valid_set, test_set, unique_characters

def process_data(path, max_size):
    print "Processing data..."

    train_x = []
    train_y = []
    valid_x = []
    valid_y = []
    test_x = []
    test_y = []

    text, length = load_text(path)

    # Truncate the length of text if max_size is specified
    if max_size is not None and max_size < length:
        text = text[:max_size]
        length = max_size

    # Specify size of different sets
    TRAIN_LEN = length * P_TRAIN
    VALID_LEN = length * (P_TRAIN + P_VALID)

    # Divide text into training, validation, and test sets
    for i in xrange(0, length-(N_IN+N_OUT), N_STEPS):
        # Split data into x, y, and chunk
        in_idx = i
        out_idx = i + N_IN
        end_idx = i + N_IN + N_OUT
        chunk_x = oils[in_idx:out_idx]
        chunk_y = oils[out_idx:end_idx]

        if i < TRAIN_LEN:
            train_x.append(text[in_idx:out_idx])
            train_y.append(text[out_idx:end_idx])
        elif i < VALID_LEN:
            valid_x.append(text[in_idx:out_idx])
            valid_y.append(text[out_idx:end_idx])
        else:
            test_x.append(text[in_idx:out_idx])
            test_y.append(text[out_idx:end_idx])

    # Make datasets
    train_x = make_sequence(train_x)
    train_y = make_target(train_y)
    valid_x = make_sequence(valid_x)
    valid_y = make_target(valid_y)
    test_x = make_sequence(test_x)
    test_y = make_target(test_y)

    train_set = (np.array(train_x), np.array(train_y))
    valid_set = (np.array(valid_x), np.array(valid_y))
    test_set = (np.array(test_x), np.array(test_y))

    print "Training Set Size: %d" % train_set[0].shape[0]
    print "Validation Set Size: %d" % valid_set[0].shape[0]
    print "Test Set Size: %d" % test_set[0].shape[0]

    return train_set, valid_set, test_set, unique_char(text)

def load_text(dataset):
    """
    Loads a PlainText file into a string without '\n'.
    Returns the string and the length of the string.
    """
    f = open(dataset, "r")
    strings = f.read().replace('\n', ' ').replace('_', '')
    f.close()

    return strings, len(strings)

def make_sequence(text):
    """
    Make a sequence of shape (N_SEQ, N_STEPS, N_IN) from a given text.
     - N_SEQ: number of sentences
     - N_STEPS: "hello" --> 4
     - N_IN: "hello" take "he" --> 2

    Turn individual characters into numbers using ord(c). This can be
    reversed by applying chr(ord(c)).
    """

    # seq, first, second = [], [], []
    # nsteps = 0
    # for i in xrange(len(text)):
    #     for idx in xrange(N_IN):
    #         second.append(ord(text[i + idx]))
    #     first.append(second)
    #     nsteps += 1
    #     second = []
    #     if (nsteps >= N_STEPS):
    #         arr.append(first)
    #         first = []
    #         nsteps = 0
    return seq

def make_target(text):
    """
    Make a target of shape (N_SEQ, N_STEPS) from a given text.

    Turn individual characters into numbers using ord(c). This can be
    reversed by applying chr(ord(c)).
    """
    targets, inner = [], []
    nsteps = 0
    for i in xrange(len(text) - N_OUT):
        for idx in xrange(N_OUT):
            inner.append(ord(text[i + idx]))
        arr.append(inner)
        inner = []
        # nsteps += 1
        # if (nsteps >= N_STEPS):
        #     arr.append(inner)
        #     if (len(arr) >= N_SEQ):
        #         break
        #     nsteps = 0
    return targets

def unique_char(text):
    """
    Return the number of unique characters in a sequence of text.
    Calculates number of output classes for softmax classification.
    """
    unique = []
    for char in text:
        if char not in unique:
            unique.append(char)
    return len(unique)

if __name__ == '__main__':
    textfile = "shakespeare"
    print "Loading data..."
    datasets = load_data(textfile, 25000)
    print "Writing datasets to %s.pkl.gz..." % textfile
    path = textfile + ".pkl.gz"
    with gzip.open(path, "wb") as file:
        file.write(cPickle.dumps(datasets))
    print "Done!"