import cPickle
import gzip
import os

"""
Loads a PlainText file into a string without '\n'.
Returns the string and the length of the string.
"""
def load_text(dataset):
    filename = open(dataset, "r")
    strings = filename.read().replace('\n', ' ').replace('_', '')
    return strings, len(strings)

"""
Make a sequence of shape (n_seq, n_steps, n_in) from a given text.
 - n_seq: number of sentences
 - n_steps: "hello" --> 4
 - n_in: "hello" take "he" --> 2

Turn individual characters into numbers using ord(c). This can be
reversed by applying chr(ord(c)).
"""
def make_sequence(text, n_steps, n_in):
    arr, first, second = [], [], []
    nsteps = 0
    for i in xrange(len(text) - n_in):
        for idx in xrange(n_in):
            second.append(ord(text[i + idx]))
        first.append(second)
        nsteps += 1
        second = []
        if (nsteps >= n_steps):
            arr.append(first)
            first = []
            nsteps = 0
    return arr

"""
Make a target of shape (n_seq, n_steps) from a given text.

Turn individual characters into numbers using ord(c). This can be
reversed by applying chr(ord(c)).
"""
def make_target(text, n_seq, n_steps, n_out):
    arr, inner = [], []
    nsteps = 0
    for i in xrange(len(text) - n_out):
        for idx in xrange(n_out):
            inner.append(ord(text[i + idx]))
        arr.append(inner)
        inner = []
        # nsteps += 1
        # if (nsteps >= n_steps):
        #     arr.append(inner)
        #     if (len(arr) >= n_seq):
        #         break
        #     nsteps = 0
    return arr

"""
Return the number of unique characters in a sequence of text.
Calculates number of output classes for softmax classification.
"""
def unique_char(text):
    unique = []
    for char in text:
        if char not in unique:
            unique.append(char)
    return len(unique)

# def load_text(dataset="shakespeare", max_size=250000, valid_portion=0.1):
#     """
#     Loads a PlainText file into a string without '\n'.
#     Returns the string and the length of the string.

#     :type dataset: String
#     :param dataset: path to dataset
#     :type max_size: int
#     :param max_size: max number of characters
#     :type valid_portion: float
#     :param valid_portion: proportion of full train set used for validation set
#     """
#     #############
#     # LOAD TEXT #
#     #############
#     path = dataset + ".pkl.gz"
#     if not os.path.exists(path):
#         f = gzip.open(path, "rb")
#         train_set, valid_set, test_set = cPickle.load(f)
#         f.close()
#         return train_set, valid_set, test_set
#     else:
#         process_data(dataset + ".txt")
#     f = open(dataset, "rb")

# def process_data(path):
#     strings = f.read().replace('\n', ' ').replace('_', '')
#     strings = strings[:max_size]
#     return strings, len(strings)
#     return len(unique)