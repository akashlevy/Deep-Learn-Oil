""" Test Long Short-Term Memory model with Keras library """
import numpy as np
import qri
import random
import time

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.core import Activation, Dense, Dropout, Reshape, TimeDistributedDense
from keras.layers.recurrent import JZS1, JZS2, JZS3
from keras.models import Sequential
from keras.optimizers import SGD

# Seed random number generator
np.random.seed(42)

# Set up input, output, and hidden layer sizes
n_in = 48
n_out = 12
n_hidden = 1000

# Load QRI data
datasets = qri.load_data("../datasets/qri.pkl.gz")

# Split into 3D datasets
datasets = [(dataset[0][:,np.newaxis], dataset[1]) for dataset in datasets]
train_set, valid_set, test_set = datasets

# Build neural network
model = Sequential()
model.add(JZS3(input_dim=n_in, output_dim=n_hidden, return_sequences=True))
model.add(JZS3(input_dim=n_hidden, output_dim=n_hidden, return_sequences=True))
model.add(JZS3(input_dim=n_hidden, output_dim=n_hidden, return_sequences=True))
model.add(TimeDistributedDense(input_dim=n_hidden, output_dim=n_out))
model.add(Reshape(n_out))
# model.add(Dense(input_dim=n_hidden, output_dim=n_out))

# Use stochastic gradient descent and compile model
sgd = SGD(lr=0.001, momentum=0.99, decay=1e-6, nesterov=True)
model.compile(loss=qri.mae_clip, optimizer=sgd)

# Use early stopping and saving as callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=10)
save_best = ModelCheckpoint("models/lstm.mdl", save_best_only=True)
callbacks = [early_stop, save_best]

# Train model
t0 = time.time()
hist = model.fit(train_set[0], train_set[1], validation_data=valid_set,
                 verbose=2, callbacks=callbacks, nb_epoch=1000, batch_size=20)
time_elapsed = time.time() - t0

# Load best model
model.load_weights("models/lstm.mdl")

# Print time elapsed and loss on testing dataset
print "\nTime elapsed: %f s" % time_elapsed
print "Testing set loss: %f" % model.test_on_batch(test_set[0], test_set[1])

# Plot training and validation loss
qri.plot_train_valid_loss(hist)

# Make predictions
qri.plot_test_predictions(model, train_set)
