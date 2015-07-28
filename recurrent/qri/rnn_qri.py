"""
RNN
@author gwtaylor
"""

import cPickle as pickle
import datetime
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import theano
import time

from collections import OrderedDict
from func import sqr_error_cost, abs_error_cost, std_abs_error
from sklearn.base import BaseEstimator
from theano import config
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import process_data

logger = logging.getLogger(__name__)
plt.ion()

mode = theano.Mode(linker='cvm')
#mode = 'DEBUG_MODE'

class RNN(object):
    """
    Recurrent Neural Network class
    Outputs real values to predict oil production based on QRI Data
    """
    def __init__(self, input, n_in, n_hidden, n_out, activation=T.tanh,
                 output_type='real', use_symbolic_softmax=False):

        self.input = input
        self.activation = activation
        self.output_type = output_type

        # when using HF, SoftmaxGrad.grad is not implemented
        # use a symbolic softmax which is slightly slower than T.nnet.softmax
        # See: http://groups.google.com/group/theano-dev/browse_thread/
        # thread/3930bd5a6a67d27a
        if use_symbolic_softmax:
            def symbolic_softmax(x):
                e = T.exp(x)
                return e / T.sum(e, axis=1).dimshuffle(0, 'x')
            self.softmax = symbolic_softmax
        else:
            self.softmax = T.nnet.softmax

        # recurrent weights as a shared variable
        W_init = np.asarray(np.random.uniform(size=(n_hidden, n_hidden),
                                              low=-.01, high=.01),
                                              dtype=theano.config.floatX)
        self.W = theano.shared(value=W_init, name='W')
        # input to hidden layer weights
        W_in_init = np.asarray(np.random.uniform(size=(n_in, n_hidden),
                                                 low=-.01, high=.01),
                                                 dtype=theano.config.floatX)
        self.W_in = theano.shared(value=W_in_init, name='W_in')

        # hidden to output layer weights
        W_out_init = np.asarray(np.random.uniform(size=(n_hidden, n_out),
                                                  low=-.01, high=.01),
                                                  dtype=theano.config.floatX)
        self.W_out = theano.shared(value=W_out_init, name='W_out')

        h0_init = np.zeros((n_hidden,), dtype=theano.config.floatX)
        self.h0 = theano.shared(value=h0_init, name='h0')

        bh_init = np.zeros((n_hidden,), dtype=theano.config.floatX)
        self.bh = theano.shared(value=bh_init, name='bh')

        by_init = np.zeros((n_out,), dtype=theano.config.floatX)
        self.by = theano.shared(value=by_init, name='by')

        self.params = [self.W, self.W_in, self.W_out, self.h0,
                       self.bh, self.by]

        # For every parameter, we maintain its last update
        # Use "momentum" to keep moving in same direction
        self.updates = {}
        for param in self.params:
            init = np.zeros(param.get_value(borrow=True).shape,
                            dtype=theano.config.floatX)
            self.updates[param] = theano.shared(init)

        # recurrent function (using tanh activation function) and linear output
        # activation function
        def step(x_t, h_tm1):
            h_t = self.activation(T.dot(x_t, self.W_in) + \
                                  T.dot(h_tm1, self.W) + self.bh)
            y_t = T.dot(h_t, self.W_out) + self.by
            return h_t, y_t

        # the hidden state `h` for the entire sequence, and the output for the
        # entire sequence `y` (first dimension is always time)
        [self.h, self.y_pred], _ = theano.scan(step,
                                               sequences=self.input,
                                               outputs_info=[self.h0, None])

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = 0
        self.L1 += abs(self.W.sum())
        self.L1 += abs(self.W_in.sum())
        self.L1 += abs(self.W_out.sum())

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = 0
        self.L2_sqr += (self.W ** 2).sum()
        self.L2_sqr += (self.W_in ** 2).sum()
        self.L2_sqr += (self.W_out ** 2).sum()

        if self.output_type == 'real':
            self.loss = lambda y: self.mse(y)
        else:
            raise NotImplementedError

    def mse(self, y):
        # error between output and target
        return T.mean((self.y_pred - y) ** 2)

    def nll_binary(self, y):
        # negative log likelihood based on binary cross entropy error
        return T.mean(T.nnet.binary_crossentropy(self.p_y_given_x, y))

    def nll_multiclass(self, y):
        # negative log likelihood based on multiclass cross entropy error
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of time steps (call it T) in the sequence
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the sequence
        over the total number of examples in the sequence ; zero one
        loss over the size of the sequence
        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        # check if y has same dimension of y_pred
        if y.ndim != self.y_out.ndim:
            raise TypeError('y should have the same shape as self.y_out',
                ('y', y.type, 'y_out', self.y_out.type))

        if self.output_type in ('binary', 'softmax'):
            # check if y is of the correct datatype
            if y.dtype.startswith('int'):
                # the T.neq operator returns a vector of 0s and 1s, where 1
                # represents a mistake in prediction
                return T.mean(T.neq(self.y_out, y))
            else:
                raise NotImplementedError()


class MetaRNN(BaseEstimator):
    def __init__(self, n_in=5, n_hidden=50, n_out=5, learning_rate=0.01,
                 n_epochs=100, L1_reg=0.00, L2_reg=0.00, learning_rate_decay=1,
                 activation='tanh', output_type='real',
                 final_momentum=0.9, initial_momentum=0.5,
                 momentum_switchover=5,
                 use_symbolic_softmax=False):
        self.n_in = int(n_in)
        self.n_hidden = int(n_hidden)
        self.n_out = int(n_out)
        self.learning_rate = float(learning_rate)
        self.learning_rate_decay = float(learning_rate_decay)
        self.n_epochs = int(n_epochs)
        self.L1_reg = float(L1_reg)
        self.L2_reg = float(L2_reg)
        self.activation = activation
        self.output_type = output_type
        self.initial_momentum = float(initial_momentum)
        self.final_momentum = float(final_momentum)
        self.momentum_switchover = int(momentum_switchover)
        self.use_symbolic_softmax = use_symbolic_softmax

        self.ready()

    def ready(self):
        # input (where first dimension is time)
        self.x = T.matrix()
        # target (where first dimension is time)
        if self.output_type == 'real':
            self.y = T.matrix(name='y', dtype=theano.config.floatX)
        else:
            raise NotImplementedError
        # initial hidden state of the RNN
        self.h0 = T.vector()
        # learning rate
        self.lr = T.scalar()

        if self.activation == 'tanh':
            activation = T.tanh
        elif self.activation == 'sigmoid':
            activation = T.nnet.sigmoid
        elif self.activation == 'relu':
            activation = lambda x: x * (x > 0)
        elif self.activation == 'cappedrelu':
            activation = lambda x: T.minimum(x * (x > 0), 6)
        else:
            raise NotImplementedError

        self.rnn = RNN(input=self.x, n_in=self.n_in,
                       n_hidden=self.n_hidden, n_out=self.n_out,
                       activation=activation, output_type=self.output_type,
                       use_symbolic_softmax=self.use_symbolic_softmax)

        if self.output_type == 'real':
            self.predict = theano.function(inputs=[self.x, ],
                                           outputs=self.rnn.y_pred,
                                           mode=mode)
        else:
            raise NotImplementedError

    def shared_dataset(self, data_xy):
        """ Load the dataset into shared variables """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))
        return shared_x, shared_y

    def __getstate__(self):
        """ Return state sequence."""
        params = self._get_params()  # parameters set in constructor
        weights = [p.get_value() for p in self.rnn.params]
        state = (params, weights)
        return state

    def _set_weights(self, weights):
        """ Set fittable parameters from weights sequence.
        Parameters must be in the order defined by self.params:
            W, W_in, W_out, h0, bh, by
        """
        i = iter(weights)

        for param in self.rnn.params:
            param.set_value(i.next())

    def __setstate__(self, state):
        """ Set parameters from state sequence.
        Parameters must be in the order defined by self.params:
            W, W_in, W_out, h0, bh, by
        """
        params, weights = state
        self.set_params(**params)
        self.ready()
        self._set_weights(weights)

    def save(self, fpath='.', fname=None):
        """ Save a pickled representation of Model state. """
        fpathstart, fpathext = os.path.splitext(fpath)
        if fpathext == '.pkl':
            # User supplied an absolute path to a pickle file
            fpath, fname = os.path.split(fpath)

        elif fname is None:
            # Generate filename based on date
            date_obj = datetime.datetime.now()
            date_str = date_obj.strftime('%Y-%m-%d-%H:%M:%S')
            class_name = self.__class__.__name__
            fname = '%s.%s.pkl' % (class_name, date_str)

        fabspath = os.path.join(fpath, fname)

        logger.info("Saving to %s ..." % fabspath)
        file = open(fabspath, 'wb')
        state = self.__getstate__()
        pickle.dump(state, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.close()

    def load(self, path):
        """ Load model parameters from path. """
        logger.info("Loading from %s ..." % path)
        file = open(path, 'rb')
        state = pickle.load(file)
        self.__setstate__(state)
        file.close()

    def fit(self, X_train, Y_train, X_test=None, Y_test=None,
            validation_frequency=100):
        """ Fit model
        Pass in X_test, Y_test to compute test error and report during
        training.
        X_train : ndarray (n_seq x n_steps x n_in)
        Y_train : ndarray (n_seq x n_steps x n_out)
        validation_frequency : int
            in terms of number of sequences (or number of weight updates)
        """
        if X_test is not None:
            assert(Y_test is not None)
            self.interactive = True
            test_set_x, test_set_y = self.shared_dataset((X_test, Y_test))
        else:
            self.interactive = False

        train_set_x, train_set_y = self.shared_dataset((X_train, Y_train))

        n_train = train_set_x.get_value(borrow=True).shape[0]
        if self.interactive:
            n_test = test_set_x.get_value(borrow=True).shape[0]

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        logger.info('... building the model')

        index = T.lscalar('index')    # index to a case
        # learning rate (may change)
        l_r = T.scalar('l_r', dtype=theano.config.floatX)
        mom = T.scalar('mom', dtype=theano.config.floatX)  # momentum

        cost = self.rnn.loss(self.y) \
            + self.L1_reg * self.rnn.L1 \
            + self.L2_reg * self.rnn.L2_sqr

        compute_train_error = theano.function(inputs=[index, ],
                                              outputs=self.rnn.loss(self.y),
                                              givens={
                                                  self.x: train_set_x[index],
                                                  self.y: train_set_y[index]},
                                              mode=mode)

        if self.interactive:
            compute_test_error = theano.function(inputs=[index, ],
                        outputs=self.rnn.loss(self.y),
                        givens={
                            self.x: test_set_x[index],
                            self.y: test_set_y[index]},
                        mode=mode)

        # compute the gradient of cost with respect to theta = (W, W_in, W_out)
        # gradients on the weights using BPTT
        gparams = []
        for param in self.rnn.params:
            gparam = T.grad(cost, param)
            gparams.append(gparam)

        updates = OrderedDict()
        for param, gparam in zip(self.rnn.params, gparams):
            weight_update = self.rnn.updates[param]
            upd = mom * weight_update - l_r * gparam
            updates[weight_update] = upd
            updates[param] = param + upd

        # compiling a Theano function `train_model` that returns the
        # cost, but in the same time updates the parameter of the
        # model based on the rules defined in `updates`
        train_model = theano.function(inputs=[index, l_r, mom],
                                      outputs=cost,
                                      updates=updates,
                                      givens={
                                          self.x: train_set_x[index],
                                          self.y: train_set_y[index]},
                                          mode=mode)

        ###############
        # TRAIN MODEL #
        ###############
        logger.info('... training')
        epoch = 0

        while (epoch < self.n_epochs):
            epoch = epoch + 1
            for idx in xrange(n_train):
                effective_momentum = self.final_momentum \
                               if epoch > self.momentum_switchover \
                               else self.initial_momentum
                example_cost = train_model(idx, self.learning_rate,
                                           effective_momentum)

                # iteration number (how many weight updates have we made?)
                # epoch is 1-based, index is 0 based
                iter = (epoch - 1) * n_train + idx + 1

                if iter % validation_frequency == 0:
                    # compute loss on training set
                    train_losses = [compute_train_error(i)
                                    for i in xrange(n_train)]
                    this_train_loss = np.mean(train_losses)

                    if self.interactive:
                        test_losses = [compute_test_error(i)
                                        for i in xrange(n_test)]
                        this_test_loss = np.mean(test_losses)

                        logger.info('epoch %i, seq %i/%i, tr loss %f '
                                    'te loss %f lr: %f' % \
                        (epoch, idx + 1, n_train,
                         this_train_loss, this_test_loss, self.learning_rate))
                    else:
                        logger.info('epoch %i, seq %i/%i, train loss %f '
                                    'lr: %f' % \
                                    (epoch, idx + 1, n_train, this_train_loss,
                                     self.learning_rate))

            self.learning_rate *= self.learning_rate_decay

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

def test_real(n_epochs=20, validation_frequency=1000):
    """ Test RNN with real-valued outputs. """
    train, valid, test = process_data.load_data()
    tseq, ttargets = train
    vseq, vtargets = valid
    test_seq, test_targets = test
    length = len(tseq)

    n_hidden = 5
    n_in = 36
    n_out = 12
    n_steps = 1
    n_seq = length

    seq = [[i] for i in tseq]
    targets = [[i] for i in ttargets]

    model = MetaRNN(n_in=n_in, n_hidden=n_hidden, n_out=n_out,
                    learning_rate=0.01, learning_rate_decay=0.99,
                    n_epochs=n_epochs, activation='relu')

    model.fit(seq, targets, validation_frequency=validation_frequency)

    test_seq = [[i] for i in test_seq]
    test_targets = [[i] for i in test_targets]
    plt.close("all")
    for idx in xrange(len(test_seq)):
        guess = model.predict(test_seq[idx])
        plot_predictions(test_seq[idx][0], test_targets[idx][0], guess[0])

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    t0 = time.time()
    test_real(1, 5000)
    print "Elapsed time: %f" % (time.time() - t0)