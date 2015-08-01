<<<<<<< HEAD
"""Test long short-term memory model with Keras library"""
=======
""" Test Long Short-Term Memory model with Keras library """
<<<<<<< HEAD
import numpy as np
import qri
import random

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.core import Activation, Dense, Dropout, Reshape
from keras.layers.embeddings import Embedding
=======
>>>>>>> 10d8f650c7d57a02ec705793037e762bc0ee1854

import numpy as np
import random
import time
import qri
from keras.callbacks import EarlyStopping, ModelCheckpoint
<<<<<<< HEAD
from keras.layers.core import Dense, Dropout
=======
from keras.layers.core import Activation, Dense, Dropout
>>>>>>> 03fdfa6a6b14ae52e464b84923254153287747b2
>>>>>>> 10d8f650c7d57a02ec705793037e762bc0ee1854
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.optimizers import SGD


# Model name
MDL_NAME = "lstm"

# Seed random number generator
np.random.seed(42)

# Set up input, output, and hidden layer sizes
n_in = 48
n_out = 12
n_hidden = 100

# Load QRI data
<<<<<<< HEAD
train_set, valid_set, test_set = qri.load_data_recurrent("../datasets/qri.pkl.gz")

# Build neural network
model = Sequential()
# model.add(Embedding(input_dim=n_in, output_dim=n_in))
# model.add(Reshape(1, 12))
model.add(LSTM(input_dim=n_in, output_dim=n_hidden, init='uniform',
               activation='relu', return_sequences=False))
# model.add(Dropout(0.5))
# model.add(LSTM(input_dim=n_hidden, output_dim=n_hidden, init='uniform',
#                activation='relu', inner_activation='sigmoid', return_sequences=False))
# model.add(Dropout(0.5))
model.add(Dense(input_dim=n_hidden, output_dim=n_out))
model.add(Activation('linear'))

# Use sgd
sgd = SGD(lr=0.01, momentum=0.99)
model.compile(loss='msle', optimizer='rmsprop')

# Early stopping and saving are callbacks
save_best = ModelCheckpoint("lstm.mdl", save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss', patience=5)
=======
datasets = qri.load_data("../datasets/qri.pkl.gz")

# Split into 3D datasets
datasets = [(dataset[0][:,:,np.newaxis], dataset[1]) for dataset in datasets]
train_set, valid_set, test_set = datasets

# Build neural network
model = Sequential()
model.add(LSTM(1, 12))
model.add(Dense(12, 12))

# Use stochastic gradient descent and compile model
sgd = SGD(lr=0.001, momentum=0.99, decay=1e-6, nesterov=True)
model.compile(loss=qri.mae_clip, optimizer=sgd)

# Use early stopping and saving as callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=10)
<<<<<<< HEAD
save_best = ModelCheckpoint("models/%s.mdl" % MDL_NAME, save_best_only=True)
=======
save_best = ModelCheckpoint("models/lstm.mdl", save_best_only=True)
>>>>>>> 03fdfa6a6b14ae52e464b84923254153287747b2
>>>>>>> 10d8f650c7d57a02ec705793037e762bc0ee1854
callbacks = [early_stop, save_best]

# Train model
t0 = time.time()
hist = model.fit(train_set[0], train_set[1], validation_data=valid_set,
                 verbose=2, callbacks=callbacks, nb_epoch=1000, batch_size=20)
time_elapsed = time.time() - t0

<<<<<<< HEAD
# Print testing loss
print "\nTrain set loss: %f" % model.test_on_batch(train_set[0], train_set[1])
print "Valid set loss: %f" % model.test_on_batch(valid_set[0], valid_set[1])
print "Test set loss: %f" % model.test_on_batch(test_set[0], test_set[1])
=======
# Load best model
model.load_weights("models/%s.mdl" % MDL_NAME)

# Print time elapsed and loss on testing dataset
test_set_loss = model.test_on_batch(test_set[0], test_set[1])
print "\nTime elapsed: %f s" % time_elapsed
<<<<<<< HEAD
print "Testing set loss: %f" % test_set_loss

# Save results
qri.save_results("results/%s.out" % MDL_NAME, time_elapsed, test_set_loss)
qri.save_history("models/%s.hist" % MDL_NAME, hist.history)
=======
print "Testing set loss: %f" % model.test_on_batch(test_set[0], test_set[1])
>>>>>>> 03fdfa6a6b14ae52e464b84923254153287747b2
>>>>>>> 10d8f650c7d57a02ec705793037e762bc0ee1854

# Plot training and validation loss
qri.plot_train_valid_loss(hist.history)

# Make predictions
qri.plot_test_predictions(model, train_set)
