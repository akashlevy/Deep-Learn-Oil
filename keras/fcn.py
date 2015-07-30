"""Test fully connected neural network with Keras"""

import random, qri
from keras.callbacks import ModelCheckpoint, EarlyStopping
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

# Use stochastic gradient descent and compile model
sgd = SGD(lr=0.01, momentum=0.99)
model.compile(loss='mae', optimizer=sgd)

# Early stopping and saving are callbacks
save_best = ModelCheckpoint("fcn.mdl", save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss', patience=5)
callbacks = [early_stop, save_best]

# Train model
hist = model.fit(train_set[0], train_set[1], validation_data=valid_set,
                 verbose=2, callbacks=callbacks, nb_epoch=100, batch_size=20)

# Print testing loss
print
print "Train set loss: %f" % model.test_on_batch(train_set[0], train_set[1])
print "Valid set loss: %f" % model.test_on_batch(valid_set[0], valid_set[1])
print "Test set loss: %f" % model.test_on_batch(test_set[0], test_set[1])

# Plot training and validation loss
qri.plot_train_valid_loss(hist)

# Make predictions
qri.plot_test_predictions(model, train_set)
