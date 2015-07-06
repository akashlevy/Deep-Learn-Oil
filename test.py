print(__doc__)

# Author: Mathieu Blondel
#         Jake Vanderplas
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def f(x):
    """ function to approximate by polynomial interpolation"""
    return x * np.sin(x)


# generate points used to plot
x_plot = np.linspace(0, 10, 100)

# generate points and keep a subset of them
x = np.linspace(0, 10, 100)
rng = np.random.RandomState(0)
rng.shuffle(x)
x = np.sort(x[:20])
y = f(x)

# create matrix versions of these arrays
X = x[:, np.newaxis]
X_plot = x_plot[:, np.newaxis]

plt.plot(x_plot, f(x_plot), label="ground truth")
plt.scatter(x, y, label="training points")

for degree in [3, 4, 5]:
    model = make_pipeline(PolynomialFeatures(degree), Ridge())
    model.fit(X, y)
    y_plot = model.predict(X_plot)
    plt.plot(x_plot, y_plot, label="degree %d" % degree)

plt.legend(loc='lower left')

plt.show()

"""A simple implementation of machine learning using minibatch SGD and early
stopping"""

import cPickle
import numpy
import theano
import theano.tensor as T    


# Model parameters
learning_rate = 0.1


# Early-stopping parameters
# TO DO: UPDATE THIS
patience = 5000 # Look at at least this many examples regardless
patience_increase = 2 # Multiply patience by this when a new best is found


# A relative improvement of this much is considered significant
improvement_threshold = 0.995


# Maximum number of epochs
# TO DO: UPDATE THIS
n_epochs = 1000


# Optimized model parameters and associated loss
best_params = None
best_validation_loss = numpy.inf


# TO DO: MAKE SHARED DATASET
loss = (params*input - label)**2


# Go through this many minibatches before checking the network on the
# validation set; in this case we check every epoch
validation_frequency = min(len(train_batches), patience/2)


done_looping = False
for epoch in xrange(1, n_epochs):
    # Report "1" for first epoch, "n_epochs" for last epoch
    for minibatch_index, train_batch in enumerate(train_batches):
        params -= learning_rate * T.grad(loss, params) # Gradient descent

        # Iteration number; we want it to start at 0
        iter = (epoch - 1) * n_train_batches + minibatch_index
        if (iter + 1) % validation_frequency == 0:
            this_validation_loss = loss_function()

            # Improve patience if loss improvement is good enough
            if this_validation_loss < best_validation_loss * improvement_threshold:
                patience = max(patience, iter * patience_increase)
            best_params = copy.deepcopy(params)
            best_validation_loss = this_validation_loss

        if patience <= iter:
            done_looping = True
            break

    if done_looping:
        break












class NeuralNet(object):
    """Class for a neural network"""
    def __init__(self, input, batch_size, output_size):
        """Initialize the neural network"""
        
        # Initialize with 0 the weights W as a matrix of shape (batch_size, output_size)
        self.W = theano.shared(value=numpy.zeros((n_in, n_out), dtype=theano.config.floatX),
                               name='W', borrow=True)
        
        # Initialize the biases b as a vector of output_size 0s
        self.b = theano.shared(value=numpy.zeros((n_out,), dtype=theano.config.floatX),
                               name='b', borrow=True)