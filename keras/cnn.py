"""Test fully connected neural network with Keras"""

import numpy as np, random, qri
from keras.layers.core import Dense, Flatten, Reshape
from keras.layers.convolutional import Convolution1D
from keras.models import Sequential
from keras.optimizers import SGD


# Seed random number generator
random.seed(42)

# Load QRI data
train_set, valid_set, test_set = qri.load_data("../datasets/qri.pkl.gz")

# Build neural network
model = Sequential()
model.add(Reshape(1, 48))
model.add(Convolution1D(input_dim=48, nb_filter=30, filter_length=12, activation="relu"))
model.add(Flatten())
model.add(Dense(input_dim=30, output_dim=12, init="uniform"))

# Use stochastic gradient descent
sgd = SGD(lr=0.01, momentum=0.99)
model.compile(loss='mae', optimizer=sgd)

# Train Model
train_set_x = np.expand_dims(train_set[0], 2)
model.fit(train_set_x, train_set[1], validation_data=valid_set, nb_epoch=100,
          batch_size=20, verbose=2)