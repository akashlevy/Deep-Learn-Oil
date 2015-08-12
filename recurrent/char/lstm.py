""" Test Long Short-Term Memory model with Keras library """
import numpy as np
import char
import random
import time

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
n_hidden = 200

# Load datasets
path = dataset + ".pkl.gz"
datasets = char.load_data(path)

# Split into 3D datasets
# train_set, valid_set, test_set, unique_char = datasets
# train_set = (train_set[0][:,np.newaxis], train_set[1])
# valid_set = (valid_set[0][:,np.newaxis], valid_set[1])
# test_set = (test_set[0][:,np.newaxis], test_set[1])
datasets = [(data[0][:,np.newaxis], data[1]) for data in datasets]

#Build neural network
model = Sequential()
model.add(Embedding(unique_char, n_in))
model.add(LSTM(input_dim=n_in, output_dim=n_hidden, activation='sigmoid',
               inner_activation='hard_sigmoid', return_sequences=True))
model.add(LSTM(input_dim=n_hidden, output_dim=n_hidden, activation='sigmoid',
               inner_activation='hard_sigmoid', return_sequences=True))
model.add(LSTM(input_dim=n_hidden, output_dim=n_hidden, activation='sigmoid',
               inner_activation='hard_sigmoid', return_sequences=False))
model.add(Dense(input_dim=n_hidden, output_dim=n_out))
model.add(Activation('sigmoid'))

# Compile using RMSprop
model.compile(loss='binary_crossentropy', optimizer='rmsprop')

# Use early stopping and saving as callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5)
save_best = ModelCheckpoint("models/%s.mdl" % dataset, save_best_only=True)
callbacks = [early_stop, save_best]

# Train model
t0 = time.time()
hist = model.fit(train_set[0], train_set[1], validation_data=valid_set,
                 verbose=2, callbacks=callbacks, nb_epoch=5, batch_size=20)
time_elapsed = time.time() - t0

# Load best model
model.load_weights("models/%s.mdl" % dataset)

# Print time elapsed and loss on testing dataset
print "\nTime elapsed: %f s" % time_elapsed
print "Testing set loss: %f" % model.test_on_batch(test_set[0], test_set[1])

# Make predictions
char.print_test_predictions(model, train_set)