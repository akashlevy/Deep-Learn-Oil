"""Testing nnet1d's convolutional neural network on QRI oil data"""

from nnet1d import NNet1D

# Build convolutional neural network with two convolutional layers and one
# fully connected layer (activation function is rectified linear units)
network = NNet1D(datafile="../datasets/qri.pkl.gz", seed=42, batch_size=25,
                 learning_rate=0.01)
network.add_conv_pool_layer(filters=100, filter_length=5, poolsize=4)
network.add_conv_pool_layer(filters=100, filter_length=4, poolsize=5)
network.add_fully_connected_layer()
network.build()

# Train neural network
for epoch in xrange(150):
    training_error, validation_error = network.train()
    output = "Epoch %d: (training error = %f, validation error = %f)"
    output %= (epoch + 1, training_error, validation_error)
    print output

# Test neural network
print "Testing error = %f" % network.test_error()

# Save model
import gzip, cPickle
with gzip.open("big_model.pkl.gz", "wb") as file:
    file.write(cPickle.dumps(network))


# DO PLOTTING
import numpy as np
import matplotlib.pyplot as plt

predictions = network.output(network.test_set_x.get_value(borrow=True))
for chunk in zip(network.test_set_x.get_value(borrow=True), network.test_set_y.get_value(borrow=True), predictions):
    # Create a figure and add a subplot with labels
    fig = plt.figure(1)
    graph = fig.add_subplot(111)
    fig.suptitle("Chunk Data", fontsize=25)
    plt.xlabel("Year", fontsize=15)
    plt.ylabel("Production", fontsize=15)
    
    # Plot the generated predictions as a blue line with round markers
    graph.plot(np.append(chunk[0], chunk[2]), "b-o", label="Generated Output")
    
    # Plot the actual predictions as a green line with round markers
    graph.plot(np.append(chunk[0], chunk[1]), "g-o", label="Actual Output")

    # Plot the data as a red line with round markers
    graph.plot(chunk[0], "r-o", label="Oil Output")

    # Add legend, resize windows, and display plot
    plt.legend()
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()