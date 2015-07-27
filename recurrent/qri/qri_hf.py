"""
This code uses the recurrent neural net implementation in rnn_qri.py.
Trains using Hessian-Free optimization.

Requires the theano-hf package:
https://github.com/boulanni/theano-hf
@author Graham Taylor
"""

import logging
import matplotlib.pyplot as plt
import numpy as np

from hf import SequenceDataset, hf_optimizer
from func import sqr_error_cost, abs_error_cost, std_abs_error
from rnn_qri import MetaRNN

import process_data

def plot_predictions(curr_seq, curr_targets, curr_guess, display_figs=True, save_figs=False,
                     output_folder="images", output_format="png"):
    """ Plots the predictions """
    # Create a figure and add a subplot with labels
    fig = plt.figure()
    graph = plt.subplot(111)
    fig.suptitle("Chunk Data", fontsize=25)
    plt.xlabel("Month", fontsize=15)
    plt.ylabel("Production", fontsize=15)
    
    # Make and display error label
    mean_abs_error = abs_error_cost(curr_targets, curr_guess).eval()
    abs_error = std_abs_error(curr_targets, curr_guess).eval()
    error = (mean_abs_error, abs_error)
    plt.title("Mean Abs Error: %f, Std: %f" % error, fontsize=10)

    # Plot the predictions as a blue line with round markers
    prediction = np.append(curr_seq, curr_guess)
    graph.plot(prediction, "b-o", label="Prediction")

    # Plot the future as a green line with round markers
    future = np.append(curr_seq, curr_targets)
    graph.plot(future, "g-o", label="Future")

    # Plot the past as a red line with round markers
    graph.plot(curr_seq, "r-o", label="Past")

    # Add legend
    plt.legend(loc="upper left")

    # Save the graphs to a folder
    if save_figs:
        filename = "%s/%04d.%s" % (output_folder, i, output_format)
        fig.savefig(filename, format=output_format)

    # Display the graph
    if display_figs:
        plt.show(block=True)

    # Clear the graph
    plt.close(fig)

def test_real(n_updates=100):
    """ Test RNN with real-valued outputs. """
    train, valid, test = process_data.load_data()
    tseq, ttargets = train
    vseq, vtargets = valid
    test_seq, test_targets = test
    length = len(tseq)

    n_hidden = 6
    n_in = 36
    n_out = 12
    n_steps = 1
    n_seq = length

    seq = [[i] for i in tseq]
    targets = [[i] for i in ttargets]

    gradient_dataset = SequenceDataset([seq, targets], batch_size=None, number_batches=100)
    cg_dataset = SequenceDataset([seq, targets], batch_size=None,
                                 number_batches=20)

    model = MetaRNN(n_in=n_in, n_hidden=n_hidden, n_out=n_out,
                    learning_rate=0.001, learning_rate_decay=0.999,
                    n_epochs=400, activation='tanh')

    opt = hf_optimizer(p=model.rnn.params, inputs=[model.x, model.y],
                       s=model.rnn.y_pred,
                       costs=[model.rnn.loss(model.y)], h=model.rnn.h)

    opt.train(gradient_dataset, cg_dataset, num_updates=n_updates)

    test_seq = [[i] for i in test_seq]
    test_targets = [[i] for i in test_targets]
    plt.close("all")
    for idx in xrange(len(test_seq)):
        guess = model.predict(test_seq[idx])
        plot_predictions(test_seq[idx][0], test_targets[idx][0], guess[0])

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


def test_softmax(n_updates=250):
    """ Test RNN with softmax outputs. """
    n_hidden = 10
    n_in = 5
    n_steps = 10
    n_seq = 100
    n_classes = 3
    n_out = n_classes  # restricted to single softmax per time step

    np.random.seed(0)
    # simple lag test
    seq = np.random.randn(n_seq, n_steps, n_in)
    targets = np.zeros((n_seq, n_steps), dtype='int32')

    thresh = 0.5
    # if lag 1 (dim 3) is greater than lag 2 (dim 0) + thresh
    # class 1
    # if lag 1 (dim 3) is less than lag 2 (dim 0) - thresh
    # class 2
    # if lag 2(dim0) - thresh <= lag 1 (dim 3) <= lag2(dim0) + thresh
    # class 0
    targets[:, 2:][seq[:, 1:-1, 3] > seq[:, :-2, 0] + thresh] = 1
    targets[:, 2:][seq[:, 1:-1, 3] < seq[:, :-2, 0] - thresh] = 2
    #targets[:, 2:, 0] = np.cast[np.int](seq[:, 1:-1, 3] > seq[:, :-2, 0])

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
    test_real(n_updates=200)
    #test_binary(multiple_out=True, n_updates=20)
    # test_softmax(n_updates=20)