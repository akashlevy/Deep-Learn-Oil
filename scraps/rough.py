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