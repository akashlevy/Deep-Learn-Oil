"""Testing nnet1d's convolutional neural network on QRI oil data"""

import numpy as np
import matplotlib.pyplot as plt
from nnet1d import NNet1D
from nnet_functions import sqr_error_cost

# List of models
models = []

# Convolutional model 1
models.append(NNet1D(datafile="../datasets/qri.pkl.gz", seed=42, batch_size=50,
                     learning_rate=0.01, momentum=0.5, cost_fn=sqr_error_cost))
models[-1].add_conv_pool_layer(filters=100, filter_length=5, poolsize=4)
models[-1].add_conv_pool_layer(filters=100, filter_length=4, poolsize=5)
models[-1].add_fully_connected_layer()
models[-1].build()

# Convolutional model 2
models.append(NNet1D(datafile="../datasets/qri.pkl.gz", seed=42, batch_size=50,
                     learning_rate=0.01, momentum=0.5, cost_fn=sqr_error_cost))
models[-1].add_conv_pool_layer(filters=100, filter_length=7, poolsize=5)
models[-1].add_conv_pool_layer(filters=100, filter_length=4, poolsize=3)
models[-1].add_fully_connected_layer()
models[-1].build()

# Convolutional model 3
models.append(NNet1D(datafile="../datasets/qri.pkl.gz", seed=42, batch_size=50,
                     learning_rate=0.01, momentum=0.5, cost_fn=sqr_error_cost))
models[-1].add_conv_pool_layer(filters=100, filter_length=9, poolsize=7)
models[-1].add_conv_pool_layer(filters=100, filter_length=2, poolsize=3)

for model in models:
    # Train neural network
    for epoch in xrange(1000):
        training_error, validation_error = model.train()
        output = "(%f, %f, %f)"
        output %= (epoch + 1, training_error, validation_error)
        print output
    
    # Test neural network
    print "Testing error = %f\n" % model.test_error()
    
# Save models
import gzip, cPickle
with gzip.open("models/theano_model.pkl.gz", "wb") as file:
    file.write(cPickle.dumps(models))

'''
# Load model
import gzip, cPickle
with gzip.open("theano_model.pkl.gz", "rb") as file:
    network = cPickle.load(file)
    
# Load test data and make prediction
x = network.test_set_x.get_value()
y = network.test_set_y.get_value()
prediction = network.output(x)

for chunk in zip(x, y, prediction):
    # Create a figure and add a subplot with labels
    fig = plt.figure(1)
    graph = fig.add_subplot(111)
    fig.suptitle("Chunk Data", fontsize=25)
    plt.title("Square Error: %f" % sqr_error_cost(chunk[1], chunk[2]).eval())
    plt.xlabel("Month", fontsize=15)
    plt.ylabel("Production", fontsize=15)
    
    # Plot the generated predictions as a blue line with round markers
    graph.plot(np.append(chunk[0], chunk[2]), "b-o", label="Generated Output")
    
    # Plot the actual predictions as a green line with round markers
    graph.plot(np.append(chunk[0], chunk[1]), "g-o", label="Actual Output")

    # Plot the data as a red line with round markers
    graph.plot(chunk[0], "r-o", label="Oil Output")

    # Add legend and display plot
    plt.legend()
    plt.show()
'''