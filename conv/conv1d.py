"""Library for 1-D convolutional neural networks"""

import cPickle
import gzip
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv


# Theano settings and random number generator
theano.config.floatX = "float32"


def load_data(dataset):
    """Loads the dataset"""
    # Unpickle from file
    with gzip.open(dataset, 'rb') as file:
        train_set, valid_set, test_set = cPickle.load(file)

    def shared_dataset(data_xy, borrow=True):
        """Loads the dataset into shared variables"""
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, shared_y

    # Load each data set
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    # Return the result
    return [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]


def relu(x):
    """Rectified linear units activation function"""
    return theano.tensor.switch(x < 0, 0, x)


def sqr_error_cost(y, output):
    """Return the average square error between output vector and y"""
    return T.mean(T.sqr(y - output))
    

class ConvNetwork(object):
    """A complete convolutional neural network"""
    def __init__(self, datasets, x, y, batch_size, layers, learning_rate):
        """Initialize network"""        
        # Split into datasets
        self.train_set_x, self.train_set_y = datasets[0]
        self.valid_set_x, self.valid_set_y = datasets[1]
        self.test_set_x, self.test_set_y = datasets[2]
        
        # Determine number of batches for each dataset
        self.n_train_batches = self.train_set_x.get_value(borrow=True).shape[0]/batch_size
        self.n_test_batches = self.test_set_x.get_value(borrow=True).shape[0]/batch_size
        self.n_valid_batches = self.valid_set_x.get_value(borrow=True).shape[0]/batch_size
        
        # Store model
        self.cost = layers[-1].cost(y)
        self.layers = layers
        self.learning_rate = learning_rate
        self.params = [param for layer in layers for param in layer.params]
        
        # Make functions
        index = T.lscalar()
        givens = {x: self.train_set_x[index * batch_size: (index + 1) * batch_size], y: self.train_set_y[index * batch_size: (index + 1) * batch_size]}
        updates = [(param_i, param_i - learning_rate * grad_i) for param_i, grad_i in zip(self.params, T.grad(self.cost, self.params))]
        self.train = theano.function([index], self.cost, updates=updates, givens=givens)
        givens = {x: self.valid_set_x[index * batch_size: (index + 1) * batch_size], y: self.valid_set_y[index * batch_size: (index + 1) * batch_size]}
        self.validate = theano.function([index], self.cost, givens = givens)
        givens = {x: self.test_set_x[index * batch_size: (index + 1) * batch_size], y: self.test_set_y[index * batch_size: (index + 1) * batch_size]}
        self.test = theano.function([index], self.cost, givens=givens)
        
    def train(self, train_steps): # EVENTUALLY INTRODUCE EARLY STOPPING
        """Train network"""
        for epoch in range(train_steps):
            costs = [train_model(i) for i in xrange(self.n_train_batches)]
            validation_losses = [validate(i) for i in xrange(self.n_valid_batches)]
            print "Epoch {}    NLL {:.2}    %err in validation set {:.1%}".format(epoch + 1, np.mean(costs), np.mean(validation_losses))
            
    def test(self):
        """Test network"""
        test_errors = [self.test_model(i) for i in range(self.n_test_batches)]
        print "test errors: {:.1%}".format(np.mean(test_errors))


class ConvPoolLayer(object):
    """Pool layer of a 1-D convolutional network"""
    def __init__(self, rng, input, input_length, batch_size, filters, filter_length, input_number=1, poolsize=1, activation_function=relu, W_bound=0.1):
        """Initialize layer"""
        # Make sure that convolution output is evenly divisible by poolsize
        assert (input_length - filter_length + 1) % poolsize == 0
        
        # Determine input, output, and filter tensor sizes
        self.input_size = (batch_size, input_number, 1, input_length)
        self.output_size = (batch_size, input_number*filters, 1, (input_length-filter_length+1)/poolsize)
        self.filter_size = (filters, input_number, 1, filter_length)
        
        # Reshape input to input size and store
        self.input = input.reshape(self.input_size)
        
        # Model parameters (weights and biases)
        self.W = theano.shared(np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=self.filter_size), dtype=theano.config.floatX), borrow=True)
        self.b = theano.shared(value=np.zeros(filter_length, dtype=theano.config.floatX), borrow=True)

        # Convolve input feature maps with filters
        conv_out = conv.conv2d(self.input, self.W, image_shape=self.input_size, filter_shape=self.filter_size)

        # Downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out, ds=(1, poolsize), ignore_border=True)
        
        # Store output
        self.output = activation_function(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # Store parameters of this layer
        self.params = [self.W, self.b]


class FullyConnectedLayer(object):
    """Fully connected layer of a 1-D convolutional network"""
    def __init__(self, rng, input, n_in, n_out, W_bound = 3, cost_function=sqr_error_cost):
        """Initialize fully connected layer"""
        # Store input and cost function
        self.input = input
        self.cost_function = cost_function
        
        # Model parameters (weights and biases)
        self.W = theano.shared(np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=(n_in, n_out)), dtype=theano.config.floatX), borrow=True)
        self.b = theano.shared(np.zeros((n_out,), dtype=theano.config.floatX), borrow=True)
        
        # Store output and params of this layer
        self.output = T.dot(input, self.W) + self.b
        self.params = [self.W, self.b]

    def cost(self, y):
        """Return the cost associated with the output"""
        return self.cost_function(y, self.output)
