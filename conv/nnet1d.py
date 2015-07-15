"""Library for 1D neural networks using Theano: supports convolutional and
fully connected layers"""

import cPickle
import gzip
import numpy as np
import theano
import theano.tensor as T
from layers1d import ConvPoolLayer, FullyConnectedLayer


# Configure floating point numbers for Theano
theano.config.floatX = "float32"
    

class NNet1D(object):
    """A neural network implemented for 1D neural networks in Theano"""
    def __init__(self, seed, datafile, batch_size, learning_rate):
        """Initialize network: seed the random number generator, load the
        datasets, and store model parameters"""
        # Store random number generator, batch size, and learning rate
        self.rng = np.random.RandomState(seed)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Initialize layers in the neural network
        self.layers = []
        
        # Input and output matrices (2D)
        self.x = T.matrix('x')
        self.y = T.matrix('y')
        
        # Split into training, validation, and testing datasets
        datasets = NNet1D.load_data(datafile)
        self.train_set_x, self.train_set_y = datasets[0]
        self.valid_set_x, self.valid_set_y = datasets[1]
        self.test_set_x, self.test_set_y = datasets[2]
        
        # Determine input and output sizes
        self.n_in = self.train_set_x.get_value(borrow=True).shape[1]
        self.n_out = self.train_set_y.get_value(borrow=True).shape[1]
        
        # Determine number of batches for each dataset
        self.n_train_batches = self.train_set_x.get_value(borrow=True).shape[0]
        self.n_train_batches /= batch_size
        self.n_valid_batches = self.valid_set_x.get_value(borrow=True).shape[0]
        self.n_valid_batches /= batch_size
        self.n_test_batches = self.test_set_x.get_value(borrow=True).shape[0]
        self.n_test_batches /= batch_size
        
    def add_conv_pool_layer(self, filters, filter_length, poolsize):
        """Add a convolutional layer to the network"""
        # If first layer, use x as input
        if len(self.layers) == 0:
            input = self.x
            input_number = 1
            input_length = self.n_in
        
        # If previous layer is convolutional, use its output as input
        elif isinstance(self.layers[-1], ConvPoolLayer):
            input = self.layers[-1].output
            input_number = self.layers[-1].output_shape[1]
            input_length = self.layers[-1].output_shape[3]
        
        # If previous layer is fully connected, use its output as input
        elif isinstance(self.layers[-1], FullyConnectedLayer):
            input = self.layers[-1].output
            input_number = 1
            input_length = self.layers[-1].output_shape
        
        # Otherwise raise error
        else:
            raise TypeError("Invalid previous layer")
            
        # Add the layer
        self.layers.append(ConvPoolLayer(rng=self.rng,
                                         input=input,
                                         input_number=input_number,
                                         input_length=input_length,
                                         batch_size=self.batch_size,
                                         filters=filters,
                                         filter_length=filter_length,
                                         poolsize=poolsize))
        
    def add_fully_connected_layer(self, output_length=None):
        """Add a fully connected layer to the network"""
        # If output_length is None, use self.n_out
        if output_length is None:
            output_length = self.n_out
        
        # If first layer, use x as input
        if len(self.layers) == 0:
            input = self.x
            input_length = self.n_in
        
        # If previous layer is convolutional, use its flattened output as input
        elif isinstance(self.layers[-1], ConvPoolLayer):
            input = self.layers[-1].output.flatten(2)
            output_shape = self.layers[-1].output_shape
            input_length = self.layers[-1].filter_shape[1] * output_shape[3]
        
        # If previous layer is fully connected, use its output as input
        elif isinstance(self.layers[-1], FullyConnectedLayer):
            input = self.layers[-1].output
            input_length = self.layers[-1].output_shape
        
        # Otherwise raise error
        else:
            raise TypeError("Invalid previous layer")
            
        # Add the layer
        self.layers.append(FullyConnectedLayer(rng=self.rng,
                                               input=input,
                                               input_length=input_length,
                                               output_length=output_length,
                                               batch_size=self.batch_size))
    
    def build(self):
        """Build the neural network from the given layers"""
        # Last layer must be fully connected and produce correct output size
        assert isinstance(self.layers[-1], FullyConnectedLayer)
        assert self.layers[-1].output_shape == self.n_out
        
        # Cost function is last layer's output cost
        self.cost = self.layers[-1].cost(self.y)
        
        # Index for batching
        i = T.lscalar()
        
        # Batching for training set
        batch_size = self.batch_size
        givens = {self.x: self.train_set_x[i*batch_size:(i+1)*batch_size],
                  self.y: self.train_set_y[i*batch_size:(i+1)*batch_size]}
        
        # Stochastic gradient descent algorithm for training function
        params = [param for layer in self.layers for param in layer.params]
        grads = T.grad(self.cost, params)
        updates = [(param_i, param_i - self.learning_rate * grad_i)
                   for param_i, grad_i in zip(params, grads)]
        
        # Make Theano training function
        self.train_model = theano.function([i], self.cost, updates=updates,
                                           givens=givens)
        
        # Batching for validation set
        givens = {self.x: self.valid_set_x[i*batch_size:(i+1)*batch_size],
                  self.y: self.valid_set_y[i*batch_size:(i+1)*batch_size]}
        
        # Make Theano validation function
        self.validate_model = theano.function([i], self.cost, givens = givens)
        
        # Batching for testing set
        givens = {self.x: self.test_set_x[i*batch_size:(i+1)*batch_size],
                  self.y: self.test_set_y[i*batch_size:(i+1)*batch_size]}
        
        # Make Theano testing function
        self.test_model = theano.function([i], self.cost, givens=givens)
        
        # Shared variables for output
        x = T.matrix()
        givens = {self.x: x} #############FIX THIS#################
        
        # Make Theano output function
        self.output = theano.function([x], self.layers[-1].output, givens=givens)

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

    def train(self):
        """Apply one training step of the network and return average training
        and validation error"""
        train_errors = [self.train_model(i)
                        for i in xrange(self.n_train_batches)]
        valid_errors = [self.validate_model(i)
                        for i in xrange(self.n_valid_batches)]
        return np.mean(train_errors), np.mean(valid_errors)
            
    def test_error(self):
        """Return average test error from the network"""
        test_errors = [self.test_model(i) for i in range(self.n_test_batches)]
        return np.mean(test_errors)
