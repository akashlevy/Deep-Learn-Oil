"""Test fully connected neural network with Keras"""

import random, qri
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import SGD


# Seed random number generator
random.seed(42)

# Load QRI data
train_set, valid_set, test_set = qri.load_data("../datasets/qri.pkl.gz")

# Build neural network
model = Sequential()
model.add(Dense(input_dim=48, output_dim=30, activation="relu"))
model.add(Dense(input_dim=30, output_dim=12))

# Use stochastic gradient descent
sgd = SGD(lr=0.01, momentum=0.99)
model.compile(loss='mae', optimizer=sgd)

# Train model
model.fit(train_set[0], train_set[1], validation_data=valid_set, nb_epoch=100,
          batch_size=20, verbose=2)