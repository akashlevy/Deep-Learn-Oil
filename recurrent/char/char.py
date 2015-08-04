"""Library for character data to work with Keras"""

import cPickle, gzip
import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T
import warnings

# Ignore warnings
warnings.simplefilter("ignore")

def load_data(filename):
    """Load datasets from a file"""
    with gzip.open(filename, "rb") as file:
        return cPickle.load(file)

def print_test_predictions(model, test_set, max_predictions=1, save_txt=False,
                          output_folder="text", output_format="txt"):
    """Plots the predictions for the first batch of the test set"""
    # Load test data and make prediction
    x = test_set[0]
    y = test_set[1]
    predictions = model.predict(x, batch_size=1, verbose=0)
    
    # Print prediction
    for i, chunk in enumerate(zip(x, y, predictions)):        
        print "============== Prediction %d ==============" % i

        # Report loss
        loss = model.test_on_batch(chunk[0][np.newaxis], chunk[1][np.newaxis])
        print "loss: %f" % loss

        # Print predictions
        prediction = chunk[2]
        print prediction

        # Print original
        future = chunk[1]
        print future
        
        # Save the txt to a folder
        if save_txt:
            filename = "%s/%04d.%s" % (output_folder, i, output_format)
            # fig.savefig(filename, format=output_format)

        # Stop predicting
        if i >= max_predictions:
            return

def plot_train_valid_loss(history):
    """Plot the training and validation error as a function of epochs"""
    # Create a figure and add a subplot with labels
    fig = plt.figure()
    graph = fig.add_subplot(111)
    fig.suptitle("Loss vs. Training Steps", fontsize=25)
    plt.xlabel("Epoch", fontsize=15)
    plt.ylabel("Loss", fontsize=15)
    
    # Plot the training error
    graph.plot(history["loss"], label="Training Set")
    
    # Plot the validation error
    graph.plot(history["val_loss"], label="Validation Set")
    
    # Add legend and display plot
    plt.legend()
    plt.show()

def print_output_graph(model, format="svg", outfile="out"):
    """Print computational graph for producing output to filename in
       specified format"""
    return theano.printing.pydotprint(model._predict, format=format,
                                      outfile=outfile)

def plot_weights(layer, cmap="gray"):
    """Plot the weight matrix"""
    for param in layer.get_weights():
        param = param.reshape(-1, layer.output_dim)
        fig = plt.figure(1)
        graph = fig.add_subplot(111)
        mat = graph.matshow(param, cmap=cmap, interpolation="none")
        fig.colorbar(mat)
        plt.show()

def save_results(filename, time_elapsed, test_set_loss):
    """Save performance of model to a file"""
    with open(filename, "w") as file:
        file.write("Time elapsed: %f s\n" % time_elapsed)
        file.write("Testing set loss: %f" % test_set_loss)

def save_history(filename, history):
    """Save history of model to a file"""
    with gzip.open(filename, "wb") as file:
        cPickle.dump(history, file)