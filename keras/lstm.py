""" Test Long Short-Term Memory model with Keras library """
import qri
import random

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD

# Seed random number generator
random.seed(42)

# Load QRI data
train_set, valid_set, test_set = qri.load_data("../datasets/qri.pkl.gz")

# Build neural network
model = Sequential()
model.add(Embedding(input_dim=48, output_dim=100))
model.add(LSTM(input_dim=48, output_dim=30, init='uniform',
               activation='relu', inner_activation='sigmoid', return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(input_dim=30, output_dim=30, init='uniform',
               activation='relu', inner_activation='sigmoid', return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(input_dim=30, output_dim=12))
model.add(Activation('relu'))

# Use sgd
sgd = SGD(lr=0.01, momentum=0.99)
model.compile(loss='mae', optimizer='sgd')

# Early stopping and saving are callbacks
save_best = ModelCheckpoint("lstm.mdl", save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss', patience=5)
callbacks = [early_stop, save_best]

# Train model
hist = model.fit(train_set[0], train_set[1], validation_data=valid_set, nb_epoch=100,
                 callbacks=callbacks, batch_size=20, verbose=2, show_accuracy=True)

# Test model
score = model.evaluate(test_set[0], test_set[1], batch_size=20)

# Print testing loss
print "\nTrain set loss: %f" % model.test_on_batch(train_set[0], train_set[1])
print "Valid set loss: %f" % model.test_on_batch(valid_set[0], valid_set[1])
print "Test set loss: %f" % model.test_on_batch(test_set[0], test_set[1])

# Plot training and validation loss
qri.plot_train_valid_loss(hist)

# Make predictions
qri.plot_test_predictions(model, train_set)