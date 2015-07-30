"""Library for 1D neural networks using Theano: supports convolutional, fully
connected, and recurrent layers"""

import copy
import cPickle
import gzip
import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T
import time
import warnings
from layers1d import ConvPoolLayer, FullyConnectedLayer, RecurrentLayer, Layer
from nnet_fns import abs_error_cost, relu


# Ignore warnings
warnings.simplefilter("ignore")


# Configure floating point numbers for Theano
theano.config.floatX = "float32"
    

class NNet1D(object):
    """A neural network implemented for 1D neural networks in Theano"""
    def __init__(self, seed, datafile, batch_size, learning_rate, momentum,
                 cost_fn=abs_error_cost):
        """Initialize network: seed the random number generator, load the
        datasets, and store model parameters"""
        # Store random number generator, batch size, learning rate and momentum
        self.rng = np.random.RandomState(seed)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        # Store cost function
        self.cost_fn = cost_fn
        
        # Initialize layers in the neural network
        self.layers = []
        
        # Input and output tensors
        self.x = T.matrix('x', theano.config.floatX)
        self.y = T.matrix('y', theano.config.floatX)
        
        # Split into training, validation, and testing datasets
        datasets = NNet1D.load_data(datafile)
        self.train_set_x, self.train_set_y = datasets[0]
        self.valid_set_x, self.valid_set_y = datasets[1]
        self.test_set_x, self.test_set_y = datasets[2]
        
        # Determine input and output sizes
        self.n_in = self.train_set_x.get_value(borrow=True).shape[1]
        self.n_out = self.train_set_y.get_value(borrow=True).shape[1]
        
        # Determine number of batches for training dataset
        self.n_train_batches = self.train_set_x.get_value(borrow=True).shape[0]
        self.n_train_batches /= batch_size

    def add_conv_pool_layer(self, filters, filter_length, poolsize,
                            activ_fn=relu, W_bound=0.01):
        """Add a convolutional layer to the network"""        
        # If first layer, use x as input
        if len(self.layers) == 0:
            new_shape = (self.batch_size, 1, 1, self.n_in)
            input = self.x.dimshuffle(0,'x','x',1)
            input_number = 1
            input_length = self.n_in
        
        # If previous layer is convolutional, use output as input
        elif isinstance(self.layers[-1], ConvPoolLayer):
            input = self.layers[-1].output
            input_number = self.layers[-1].filter_shape[0]
            input_length = self.layers[-1].output_length
        
        # If previous layer is fully connected/recurrent, use output as input
        elif isinstance(self.layers[-1], FullyConnectedLayer):
            new_shape = (1, 1, 1, self.layers[-1].output_shape[0])
            input = self.layers[-1].output.reshape(new_shape)
            input_number = 1
            input_length = self.layers[-1].output_shape[0]
        
        # Otherwise raise error
        else:
            raise TypeError("Invalid previous layer")
            
        # Add the layer
        layer = ConvPoolLayer(self.rng, input, input_length, filters,
                              filter_length, input_number, poolsize, activ_fn,
                              W_bound)
        self.layers.append(layer)

    def add_fully_connected_layer(self, output_length=None, activ_fn=None,
                                  W_bound=0.01):
        """Add a fully connected layer to the network"""        
        # If output_length is None, use self.n_out
        if output_length is None:
            output_length = self.n_out
        
        # If first layer, use x as input
        if len(self.layers) == 0:
            input = self.x
            input_length = self.n_in
        
        # If previous layer is convolutional, use flattened output as input
        elif isinstance(self.layers[-1], ConvPoolLayer):
            input = self.layers[-1].output.flatten(2)
            input_length = self.layers[-1].filter_shape[1]
            input_length *= self.layers[-1].output_length
        
        # If previous layer is fully connected/recurrent, use output as input
        elif isinstance(self.layers[-1], FullyConnectedLayer):
            input = self.layers[-1].output
            input_length = self.layers[-1].output_length
        
        # Otherwise raise error
        else:
            raise TypeError("Invalid previous layer")
            
        # Add the layer
        layer = FullyConnectedLayer(self.rng, input, input_length,
                                    output_length, activ_fn, self.cost_fn,
                                    W_bound)
        self.layers.append(layer)

    def add_recurrent_layer(self, output_length=None, activ_fn=None,
                            W_bound=0.01):
        """Add a fully connected layer to the network"""        
        # If output_length is None, use self.n_out
        if output_length is None:
            output_length = self.n_out
        
        # If first layer, use x as input
        if len(self.layers) == 0:
            input = self.x
            input_length = self.n_in
        
        # If previous layer is convolutional, use flattened output as input
        elif isinstance(self.layers[-1], ConvPoolLayer):
            input = self.layers[-1].output.flatten(2)
            input_length = self.layers[-1].filter_shape[1]
            input_length *= self.layers[-1].output_length
        
        # If previous layer is fully connected/recurrent, use its output as input
        elif isinstance(self.layers[-1], FullyConnectedLayer):
            input = self.layers[-1].output
            input_length = self.layers[-1].output_length
        
        # Otherwise raise error
        else:
            raise TypeError("Invalid previous layer")
        
        # Add layer
        layer = RecurrentLayer(self.rng, input, input_length,
                               output_length, activ_fn, W_bound)
        self.layers.append(layer)

    def build(self):
        """Build the neural network from the given layers"""        
        # Last layer must be fully connected and produce correct output size
        assert isinstance(self.layers[-1], FullyConnectedLayer)
        assert self.layers[-1].output_length == self.n_out
        
        # Cost function is last layer's output cost
        self.cost = self.layers[-1].cost(self.y)
        
        # Keep a count of the number of training steps, train/valid errors
        self.epochs = 0
        self.train_errors = []
        self.valid_errors = []
        
        # Index for batching
        i = T.lscalar()
        
        # Batching for training set
        batch_size = self.batch_size
        givens = {self.x: self.train_set_x[i*batch_size:(i+1)*batch_size],
                  self.y: self.train_set_y[i*batch_size:(i+1)*batch_size]}
        
        # Stochastic gradient descent algorithm for training function
        params = [param for layer in self.layers for param in layer.params]
        updates = self.gradient_updates_momentum(params)
        
        # Make Theano training function
        self.train_batch = theano.function([i], updates=updates, givens=givens)
        
        # Shared variables for output and error functions
        x = T.matrix()
        y = T.matrix()
        output = self.layers[-1].output
        
        # Make Theano output and error functions
        givens = {self.x: x}
        self.output = theano.function([x], output, givens=givens)
        givens = {self.x: x, self.y: y}
        self.error = theano.function([x, y], self.cost, givens=givens)
        givens = {self.x: self.train_set_x, self.y: self.train_set_y}
        self.train_error = theano.function([], self.cost, givens=givens)
        givens = {self.x: self.valid_set_x, self.y: self.valid_set_y}
        self.valid_error = theano.function([], self.cost, givens=givens)
        givens = {self.x: self.test_set_x, self.y: self.test_set_y}
        self.test_error = theano.function([], self.cost, givens=givens)

    def gradient_updates_momentum(self, params):
        """Return the updates necessary to implement momentum"""
        updates = []
        for param in params:
            # Update parameter
            param_update = theano.shared(param.get_value()*0.,
                                         broadcastable=param.broadcastable)
            updates.append((param, param - self.learning_rate*param_update))
            
            # Store gradient with exponential decay
            grad = T.grad(self.cost, param)
            updates.append((param_update,
                            self.momentum*param_update +
                            (1 - self.momentum)*grad))
            
        # Return the updates
        return updates

    @staticmethod
    def load_data(filename):
        """Load the datasets from file with filename"""
        # Unpickle raw datasets from file as numpy arrays
        with gzip.open(filename, 'rb') as file:
            train_set, valid_set, test_set = cPickle.load(file)
    
        def shared_dataset(data_xy, borrow=True):
            """Load the dataset data_xy into shared variables"""
            # Split into input and output
            data_x, data_y = data_xy
            
            # Store as numpy arrays with Theano data types
            shared_x_array = np.asarray(data_x, dtype=theano.config.floatX)
            shared_y_array = np.asarray(data_y, dtype=theano.config.floatX)
            
            # Create Theano shared variables
            shared_x = theano.shared(shared_x_array, borrow=borrow)
            shared_y = theano.shared(shared_y_array, borrow=borrow)
            
            # Return shared variables
            return shared_x, shared_y
    
        # Return the resulting shared variables
        return [shared_dataset(train_set), shared_dataset(valid_set),
                shared_dataset(test_set)]

    @classmethod
    def load_model(cls, filename):
        """Load a model from a file and return the NNet1D object associated
        with it"""
        with gzip.open(filename, "rb") as file:
            return cPickle.load(file)

    def plot_test_predictions(self, display_figs=True, save_figs=False,
                              output_folder="images", output_format="png"):
        """Plots the predictions for the first batch of the test set"""
        # Load test data and make prediction
        x = self.test_set_x.get_value(borrow=True)
        y = self.test_set_y.get_value(borrow=True)
        prediction = self.output(x)

        # Plot each chunk with its prediction
        for i, chunk in enumerate(zip(x, y, prediction)):
            # Create a figure and add a subplot with labels
            fig = plt.figure(i)
            graph = fig.add_subplot(111)
            fig.suptitle("Chunk Data", fontsize=25)
            plt.xlabel("Month", fontsize=15)
            plt.ylabel("Production", fontsize=15)

            # Make and display error label
            mean_cost = abs_error_cost(chunk[1], chunk[2]).eval()
            plt.title("Mean Cost: %f" % mean_cost, fontsize=10)

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

    def plot_train_valid_error(self, model_name=""):
        """Plot the training and validation error as a function of epochs.
        Return the graph used (to allow plotting multiple curves)"""
        # Create a figure and add a subplot with labels
        fig = plt.figure(1)
        graph = fig.add_subplot(111)
        fig.suptitle("Error vs. Training Steps", fontsize=25)
        plt.xlabel("Epoch", fontsize=15)
        plt.ylabel("Absolute Error", fontsize=15)
        
        # Plot the training error
        graph.plot(self.train_errors, label="Training Set " + model_name)
        
        # Plot the validation error
        graph.plot(self.valid_errors, label="Validation Set " + model_name)
        
        # Add legend and display plot
        plt.legend()
        plt.show()

    def print_output_graph(self, outfile, format="svg"):
        """Print computational graph for producing output to filename in
        specified format"""
        return theano.printing.pydotprint(self.output, format=format,
                                          outfile=outfile)

    def save_model(self, filename):
        """Save the model to a file"""
        with gzip.open(filename, "wb") as file:
            file.write(cPickle.dumps(self))

    def train(self):
        """Apply one training step of the network and return average training
        and validation error"""
        self.epochs += 1
        for i in xrange(self.n_train_batches):
            self.train_batch(i)
        mean_train_error = self.train_error()
        mean_valid_error = self.valid_error()
        self.train_errors.append(mean_train_error)
        self.valid_errors.append(mean_valid_error)
        return mean_train_error, mean_valid_error

    def train_early_stopping(self, patience=15, improve_thresh=0.00001,
                             min_epochs=0, max_epochs=99999, print_error=True):
        """Train the model with early stopping based on validation error.
        Return the time elapsed"""
        # Start timer
        start_time = time.time()
        
        # Early stopping bests
        best_epochs = 0
        best_valid_error = np.inf

        # Train model
        while self.epochs < max_epochs:
            # Run training step on model
            train_error, valid_error = self.train()
            
            # Only check bests if past min_epochs
            if self.epochs > min_epochs:
                # If lower validation error, record new best
                if valid_error + improve_thresh < best_valid_error:
                    best_epochs = self.epochs
                    best_valid_error = valid_error
                    
                # If patience exceeded, done training
                if best_epochs + patience < self.epochs:
                    break
            
            # Print epoch, training error and validation error
            if print_error:
                errors = (self.epochs, train_error, valid_error)
                print "(%s, %s, %s)" % errors
        
        # Stop timer
        end_time = time.time()
        
        # Test neural network and stop timer
        if print_error:
            print "Testing error = %s\n" % self.test_error()
            print "Time elapsed: %f" % (end_time - start_time)
            
        # Return time elapsed
        return end_time - start_time
