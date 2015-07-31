"""Library for QRI data to work with Keras"""

import cPickle, gzip
import matplotlib.pyplot as plt
import numpy as np
import theano


def load_data(filename):
    """Load datasets from a file"""
    with gzip.open(filename, "rb") as file:
        return cPickle.load(file)

def load_data_recurrent(filename, timesteps=1):
    """ Load datasets into 3D arrays from a file """
    def make3d(data):
        seq = []
        for i in xrange(0, len(data), timesteps):
            seq.append([data[i]])
        return seq

    train_set, valid_set, test_set = load_data(filename)

    train_x = make3d(train_set[0])
    valid_x = make3d(valid_set[0])
    test_x = make3d(test_set[0])

     # Make datasets
    train_set = (np.array(train_x), train_set[1])
    valid_set = (np.array(valid_x), valid_set[1])
    test_set = (np.array(test_x), test_set[1])

    return train_set, valid_set, test_set

def plot_test_predictions(model, test_set, display_figs=True, save_figs=False,
                          output_folder="images", output_format="png"):
    """Plots the predictions for the first batch of the test set"""
    # Load test data and make prediction
    x = test_set[0]
    y = test_set[1]
    predictions = model.predict(x, batch_size=1, verbose=0)
    
    # Plot each chunk with its prediction
    for i, chunk in enumerate(zip(x, y, predictions)):        
        # Create a figure and add a subplot with labels
        fig = plt.figure()
        graph = fig.add_subplot(111)
        fig.suptitle("Chunk Data", fontsize=25)
        plt.xlabel("Month", fontsize=15)
        plt.ylabel("Production", fontsize=15)

        # Make and display error label
        x2d = np.reshape(chunk[0], (1, 48))
        y2d = np.reshape(chunk[1], (1, 12))
        loss = model.test_on_batch(x2d, y2d)
        plt.title("Loss: %f" % loss, fontsize=10)

        # Plot the predictions as a blue line with round markers
        prediction = np.append(chunk[0], chunk[2])
        graph.plot(prediction, "b-o", label="Prediction")

        # Plot the future as a green line with round markers
        future = np.append(chunk[0], chunk[1])
        graph.plot(future, "g-o", label="Future")

        # Plot the past as a red line with round markers
        past = chunk[0]
        graph.plot(past, "r-o", label="Past")

        # Add legend
        plt.legend(loc="upper left")

        # Save the graphs to a folder
        if save_figs:
            filename = "%s/%04d.%s" % (output_folder, i, output_format)
            fig.savefig(filename, format=output_format)

        # Display the graph
        if display_figs:
            plt.show()
        
        # Clear the graph
        plt.close(fig)


def plot_train_valid_loss(hist):
    """Plot the training and validation error as a function of epochs"""
    # Create a figure and add a subplot with labels
    fig = plt.figure()
    graph = fig.add_subplot(111)
    fig.suptitle("Loss vs. Training Steps", fontsize=25)
    plt.xlabel("Epoch", fontsize=15)
    plt.ylabel("Loss", fontsize=15)
    
    # Plot the training error
    graph.plot(hist.history["loss"], label="Training Set")
    
    # Plot the validation error
    graph.plot(hist.history["val_loss"], label="Validation Set")
    
    # Add legend and display plot
    plt.legend()
    plt.show()


def print_output_graph(model, format="svg", outfile="out"):
    """Print computational graph for producing output to filename in
       specified format"""
    return theano.printing.pydotprint(model._predict, format=format,
                                      outfile=outfile)
