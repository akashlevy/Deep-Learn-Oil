"""Test convolutional model with Keras"""

import numpy as np
import random
import time
import qri
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.models import Sequential
from keras.optimizers import SGD


# Model name
MDL_NAME = "cnn"

# Seed random number generator
np.random.seed(42)

# Load QRI data
datasets = qri.load_data("../datasets/qri.pkl.gz")

# Split into 3D datasets
datasets = [(dataset[0][:,:,np.newaxis], dataset[1]) for dataset in datasets]
train_set, valid_set, test_set = datasets

# Build neural network
model = Sequential()
model.add(Convolution1D(1, 100, 13, activation="relu"))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(3600, 12))

# Use stochastic gradient descent and compile model
sgd = SGD(lr=0.001, momentum=0.99, decay=1e-6, nesterov=True)
model.compile(loss=qri.mae_clip, optimizer=sgd)

# Use early stopping and saving as callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=10)
save_best = ModelCheckpoint("models/%s.mdl" % MDL_NAME, save_best_only=True)
callbacks = [early_stop, save_best]

# Train model
t0 = time.time()
hist = model.fit(train_set[0], train_set[1], validation_data=valid_set,
                 verbose=2, callbacks=callbacks, nb_epoch=1000, batch_size=20)
time_elapsed = time.time() - t0

# Load best model
model.load_weights("models/%s.mdl" % MDL_NAME)

# Print time elapsed and loss on testing dataset
test_set_loss = model.test_on_batch(test_set[0], test_set[1])
print "\nTime elapsed: %f s" % time_elapsed
print "Testing set loss: %f" % test_set_loss

# Save results
qri.save_results("results/%s.out" % MDL_NAME, time_elapsed, test_set_loss)
qri.save_history("models/%s.hist" % MDL_NAME, hist.history)

# Plot training and validation loss
qri.plot_train_valid_loss(hist.history)

# Make predictions
qri.plot_test_predictions(model, train_set)
