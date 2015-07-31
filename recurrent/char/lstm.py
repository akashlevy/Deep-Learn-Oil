""" Test Long Short-Term Memory model with Keras library """
import numpy as np
import char
import random

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.core import Activation, Dense, Dropout, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.optimizers import SGD

# Choose text
dataset = "shakespeare"

# Set up input and output sizes
n_in = 90
n_out = 10
n_steps = 1
n_classes = unique_char
n_seq = length

# Load datasets
path = dataset + ".pkl.gz"
train_set, valid_set, test_set = char.load_data(path)

#Build neural network
model = Sequential()
model.add(Embedding(input_dim=n_in, output_dim=n_out)
model.add(LSTM(input_dim=n_in, output_dim=n_hidden, activation='sigmoid',
               inner_activation='hard_sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(128, 1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop')

model.fit(X_train, Y_train, batch_size=16, nb_epoch=10)
score = model.evaluate(X_test, Y_test, batch_size=16)