"""Test loading a neural network"""

import nnet1d

# Load fully connected model
model = nnet1d.NNet1D.load_model("models/fcn.mdl")
print "FCN:"
print "Training error = %s" % model.train_error()
print "Validation error = %s" % model.valid_error()
print "Test error = %s" % model.test_error()

# Load convolutional model
model = nnet1d.NNet1D.load_model("models/cnn.mdl")
print "CNN:"
print "Training error = %s" % model.train_error()
print "Validation error = %s" % model.valid_error()
print "Test error = %s" % model.test_error()

# Load recurrent model
model = nnet1d.NNet1D.load_model("models/rnn.mdl")
print "RNN:"
print "Training error = %s" % model.train_error()
print "Validation error = %s" % model.valid_error()
print "Test error = %s" % model.test_error()
