"""
This code uses the recurrent neural net implementation in rnn.py
but trains it using Hessian-Free optimization.
It requires the theano-hf package:
https://github.com/boulanni/theano-hf
@author Graham Taylor
"""
import logging
import matplotlib.pyplot as plt
import numpy as np

from char_rnn import MetaRNN
from hf import SequenceDataset, hf_optimizer

import process_text

def test_real(n_updates=100):
    """ Test RNN with real-valued outputs. """
    n_hidden = 10
    n_in = 5
    n_out = 3
    n_steps = 2
    n_seq = 100

    np.random.seed(0)
    # simple lag test
    seq = np.random.randn(n_seq, n_steps, n_in)

    targets = np.zeros((n_seq, n_steps, n_out))
    # targets[:, 1:, 0] = seq[:, :-1, 3]  # delayed 1
    # targets[:, 1:, 1] = seq[:, :-1, 2]  # delayed 1
    # targets[:, 2:, 2] = seq[:, :-2, 0]  # delayed 2

    # targets += 0.01 * np.random.standard_normal(targets.shape)

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
    text, length = process_text.load_text(dataset)
    n_hidden = 10
    n_in = 25
    n_steps = 130
    n_seq = length / (n_steps * n_in)
    n_classes = 130 # process_text.unique_char(text) # alphanum, '.', ',', '?', '!', '\'', '"', ':', ';', ' ', '\n', '\t', '*'
    n_out = n_classes # restricted to single softmax per time step

    seq = np.asarray(process_text.make_sequence(text, n_steps, n_in), dtype='int32')
    targets = np.asarray(process_text.make_target(text, n_seq, n_steps, n_out), dtype='int32')

    # SequenceDataset wants a list of sequences
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

    for seq_num in xrange(10):
        guess = model.predict(seq[seq_num])
        print "prediction: ", chr(guess[0])

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # test_real(n_updates=20)
    #test_binary(multiple_out=True, n_updates=20)
    test_softmax("sherlock.txt", n_updates=100)
