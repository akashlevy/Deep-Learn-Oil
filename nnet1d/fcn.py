"""Testing nnet1d's convolutional neural network on QRI oil data"""

import copy
import matplotlib.pyplot as plt
import numpy as np
import theano.tensor as T
from nnet1d import NNet1D
from nnet_functions import relu, abs_error_cost

# List of models
models = []

# Fully connected model 1
models.append(NNet1D(datafile="../datasets/qri.pkl.gz", seed=42, batch_size=300,
                     learning_rate=0.01, momentum=0, cost_fn=abs_error_cost))
# models[-1].add_fully_connected_layer(output_length=32, activ_fn=T.tanh)
# models[-1].add_fully_connected_layer(output_length=28, activ_fn=T.tanh)
# models[-1].add_fully_connected_layer(output_length=24, activ_fn=T.tanh)
# models[-1].add_fully_connected_layer(output_length=20, activ_fn=T.tanh)
# models[-1].add_fully_connected_layer(output_length=16, activ_fn=T.tanh)
models[-1].add_fully_connected_layer()
models[-1].build()

# Early stopping parameters
patience = 15
min_epochs = 100
max_epochs = 50000

# Train models
for model in models:
    # Early stopping bests
    best_model = None
    best_validation_error = np.inf
    best_validation_index = 0
    
    # Train model
    for index in xrange(max_epochs):
        training_error, validation_error = model.train()
        
        # Only check bests if past min_epochs
        if index > min_epochs:
            # If lower validation error, record new best
            if validation_error < best_validation_error:
                best_model = copy.deepcopy(model)
                best_validation_error = validation_error
                best_validation_index = index
                
            # If patience exceeded, done training
            if best_validation_index + patience < index:
                break
        
        # Print epoch, training error and validation error for progress record
        print "(%s, %s, %s)" % (index + 1, training_error, validation_error)
    
    # Test neural network
    print "Testing error = %s\n" % model.test_error()
    
    # Replace old model with best model
    model = best_model

# Save models
import gzip, cPickle
with gzip.open("models/simple_model.pkl.gz", "wb") as file:
    file.write(cPickle.dumps(models))

# Load model
import gzip, cPickle
with gzip.open("models/simple_model.pkl.gz", "rb") as file:
    models = cPickle.load(file)

# Load test data and make prediction
x = models[-1].test_set_x.get_value()
y = models[-1].test_set_y.get_value()
prediction = models[-1].output(x)

for chunk in zip(x, y, prediction):
    # Create a figure and add a subplot with labels
    fig = plt.figure(1)
    graph = fig.add_subplot(111)
    fig.suptitle("Chunk Data", fontsize=25)
    mean_abs_error = abs_error_cost(chunk[1], chunk[2]).eval()
    plt.title("Mean Abs Error: %f" % mean_abs_error, fontsize=10)
    plt.xlabel("Month", fontsize=15)
    plt.ylabel("Production", fontsize=15)
    
    # Plot the generated predictions as a blue line with round markers
    graph.plot(np.append(chunk[0], chunk[2]), "b-o", label="Generated Output")
    
    # Plot the actual predictions as a green line with round markers
    graph.plot(np.append(chunk[0], chunk[1]), "g-o", label="Actual Output")

    # Plot the data as a red line with round markers
    graph.plot(chunk[0], "r-o", label="Oil Output")

    # Add legend and display plot
    plt.legend(loc="upper left")
    plt.show()
