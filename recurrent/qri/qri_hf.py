"""
This code uses the recurrent neural net implementation in rnn.py
but trains it using Hessian-Free optimization.
It requires the theano-hf package:
https://github.com/boulanni/theano-hf
@author Graham Taylor
"""

import numpy as np
import matplotlib.pyplot as plt
import logging

from rnn_mnist import MetaRNN
from hf import SequenceDataset, hf_optimizer

"""
Loads a PlainText file into a string without '\n'.
Returns the string and the length of the string.
"""
def load_text(dataset):
    filename = open(dataset, "r")
    strings = filename.read().replace('\n', ' ').replace('_', '')
    return strings, len(strings)

"""
Make a sequence of shape (n_seq, n_steps, n_in) from a given text
 - n_seq: number of sentences
 - n_steps: "hello" --> 4
 - n_in: "hello" take "he" --> 2

Turn individual characters into numbers using ord(c). This can be
reversed by applying chr(ord(c)).
"""
def make_sequence(text, n_steps, n_in):
    arr, first, second = [], [], []
    nsteps, nin = 0, 0
    for char in text:
        second.append(ord(char))
        nin += 1
        if (nin >= n_in):
            first.append(second)
            nsteps += 1
            second = []
            nin = 0
        if (nsteps >= n_steps):
            arr.append(first)
            first = []
            nsteps = 0
    return arr

"""
Make a target of shape (n_seq, n_steps) from a given text
 - n_seq: number of sentences
 - n_steps: "hello" --> 4

Turn individual characters into numbers using ord(c). This can be
reversed by applying chr(ord(c)).
"""
def make_target(text, n_seq, n_steps):
    arr, inner = [], []
    nsteps = 0
    for char in text:
        inner.append(ord(char))
        nsteps += 1
        if (nsteps >= n_steps):
            arr.append(inner)
            if (len(arr) >= n_seq):
                break
            inner = []
            nsteps = 0
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

def test_real(n_updates=100):
    """ Test RNN with real-valued outputs. """
    n_hidden = 10
    n_in = 5
    n_out = 3
    n_steps = 10
    n_seq = 1000

    np.random.seed(0)
    # simple lag test
    seq = np.random.randn(n_seq, n_steps, n_in)

    targets = np.zeros((n_seq, n_steps, n_out))
    targets[:, 1:, 0] = seq[:, :-1, 3]  # delayed 1
    targets[:, 1:, 1] = seq[:, :-1, 2]  # delayed 1
    targets[:, 2:, 2] = seq[:, :-2, 0]  # delayed 2

    targets += 0.01 * np.random.standard_normal(targets.shape)

    # SequenceDataset wants a list of sequences
    # this allows them to be different lengths, but here they're not
    seq = [i for i in seq]
    targets = [i for i in targets]

    gradient_dataset = SequenceDataset([seq, targets], batch_size=None,
                                       number_batches=100)
    cg_dataset = SequenceDataset([seq, targets], batch_size=None,
                                 number_batches=20)

    model = MetaRNN(n_in=n_in, n_hidden=n_hidden, n_out=n_out,
                    activation='tanh')

    opt = hf_optimizer(p=model.rnn.params, inputs=[model.x, model.y],
                       s=model.rnn.y_pred,
                       costs=[model.rnn.loss(model.y)], h=model.rnn.h)

    opt.train(gradient_dataset, cg_dataset, num_updates=n_updates)

    plt.close('all')
    fig = plt.figure()
    ax1 = plt.subplot(211)
    plt.plot(seq[0])
    ax1.set_title('input')
    ax2 = plt.subplot(212)
    true_targets = plt.plot(targets[0])

    guess = model.predict(seq[0])
    guessed_targets = plt.plot(guess, linestyle='--')
    for i, x in enumerate(guessed_targets):
        x.set_color(true_targets[i].get_color())
    ax2.set_title('solid: true output, dashed: model output')


def test_binary(multiple_out=False, n_updates=250):
    """ Test RNN with binary outputs. """
    n_hidden = 10
    n_in = 5
    if multiple_out:
        n_out = 2
    else:
        n_out = 1
    n_steps = 10
    n_seq = 100

    np.random.seed(0)
    # simple lag test
    seq = np.random.randn(n_seq, n_steps, n_in)
    targets = np.zeros((n_seq, n_steps, n_out), dtype='int32')

    # whether lag 1 (dim 3) is greater than lag 2 (dim 0)
    targets[:, 2:, 0] = np.cast[np.int32](seq[:, 1:-1, 3] > seq[:, :-2, 0])

    if multiple_out:
        # whether product of lag 1 (dim 4) and lag 1 (dim 2)
        # is less than lag 2 (dim 0)
        targets[:, 2:, 1] = np.cast[np.int32](
            (seq[:, 1:-1, 4] * seq[:, 1:-1, 2]) > seq[:, :-2, 0])

    # SequenceDataset wants a list of sequences
    # this allows them to be different lengths, but here they're not
    seq = [i for i in seq]
    targets = [i for i in targets]

    gradient_dataset = SequenceDataset([seq, targets], batch_size=None,
                                       number_batches=500)
    cg_dataset = SequenceDataset([seq, targets], batch_size=None,
                                 number_batches=100)

    model = MetaRNN(n_in=n_in, n_hidden=n_hidden, n_out=n_out,
                    activation='tanh', output_type='binary')

    # optimizes negative log likelihood
    # but also reports zero-one error
    opt = hf_optimizer(p=model.rnn.params, inputs=[model.x, model.y],
                       s=model.rnn.y_pred,
                       costs=[model.rnn.loss(model.y),
                              model.rnn.errors(model.y)], h=model.rnn.h)

    # using settings of initial_lambda and mu given in Nicolas' RNN example
    # seem to do a little worse than the default
    opt.train(gradient_dataset, cg_dataset, num_updates=n_updates)

    seqs = xrange(10)

    plt.close('all')
    for seq_num in seqs:
        fig = plt.figure()
        ax1 = plt.subplot(211)
        plt.plot(seq[seq_num])
        ax1.set_title('input')
        ax2 = plt.subplot(212)
        true_targets = plt.step(xrange(n_steps), targets[seq_num], marker='o')

        guess = model.predict_proba(seq[seq_num])
        guessed_targets = plt.step(xrange(n_steps), guess)
        plt.setp(guessed_targets, linestyle='--', marker='d')
        for i, x in enumerate(guessed_targets):
            x.set_color(true_targets[i].get_color())
        ax2.set_ylim((-0.1, 1.1))
        ax2.set_title('solid: true output, dashed: model output (prob)')


def test_softmax(dataset, n_updates=250):
    """ Test RNN with softmax outputs. """
    text, length = load_text(dataset)
    n_hidden = 10
    n_in = 5
    n_steps = 10
    n_seq = length / (n_in * n_steps)
    n_classes = unique_char(text) + 50 # alphanum, '.', ',', '?', '!', '\'', '"', ':', ';', ' ', '\n', '\t', '*'
    n_out = n_classes # restricted to single softmax per time step

    seq = np.asarray(make_sequence(text, n_steps, n_in))
    targets = np.asarray(make_target(text, n_seq, n_steps))
    
    # thresh = 0.5
    # # if lag 1 (dim 3) is greater than lag 2 (dim 0) + thresh
    # # class 1
    # # if lag 1 (dim 3) is less than lag 2 (dim 0) - thresh
    # # class 2
    # # if lag 2(dim0) - thresh <= lag 1 (dim 3) <= lag2(dim0) + thresh
    # # class 0
    # targets[:, 2:][seq[:, 1:-1, 3] > seq[:, :-2, 0] + thresh] = 1
    # targets[:, 2:][seq[:, 1:-1, 3] < seq[:, :-2, 0] - thresh] = 2
    # #targets[:, 2:, 0] = np.cast[np.int](seq[:, 1:-1, 3] > seq[:, :-2, 0])

    # SequenceDataset wants a list of sequences
    # this allows them to be different lengths, but here they're not
    seq = [i for i in seq]
    targets = [i for i in targets]

    gradient_dataset = SequenceDataset([seq, targets], batch_size=None,
                                       number_batches=500)
    cg_dataset = SequenceDataset([seq, targets], batch_size=None,
                                 number_batches=100)

    model = MetaRNN(n_in=n_in, n_hidden=n_hidden, n_out=n_out,
                    activation='tanh', output_type='softmax',
                    use_symbolic_softmax=True)

    # optimizes negative log likelihood
    # but also reports zero-one error
    opt = hf_optimizer(p=model.rnn.params, inputs=[model.x, model.y],
                       s=model.rnn.y_pred,
                       costs=[model.rnn.loss(model.y),
                              model.rnn.errors(model.y)], h=model.rnn.h)

    # using settings of initial_lambda and mu given in Nicolas' RNN example
    # seem to do a little worse than the default
    opt.train(gradient_dataset, cg_dataset, num_updates=n_updates)

    seqs = xrange(10)

    plt.close('all')
    for seq_num in seqs:
        fig = plt.figure()
        ax1 = plt.subplot(211)
        plt.plot(seq[seq_num])
        ax1.set_title('input')

        ax2 = plt.subplot(212)
        # blue line will represent true classes
        true_targets = plt.step(xrange(n_steps), targets[seq_num], marker='o')

        # show probabilities (in b/w) output by model
        guess = model.predict_proba(seq[seq_num])
        guessed_probs = plt.imshow(guess.T, interpolation='nearest',
                                   cmap='gray')
        ax2.set_title('blue: true class, grayscale: probs assigned by model')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    #test_real(n_updates=20)
    #test_binary(multiple_out=True, n_updates=20)
    test_softmax("pride_and_prejudice.txt", n_updates=10)