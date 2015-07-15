"""Layer classes of a 1-D neural network"""

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from nnet_functions import relu, sqr_error_cost


class ConvPoolLayer(object):
    """Convolutional layer of a 1-D neural network"""
    def __init__(self, rng, input, input_length, batch_size, filters,
                 filter_length, input_number=1, poolsize=1,
                 activation_function=relu, W_bound=0.1):
        """Initialize layer"""
        # Make sure that convolution output is evenly divisible by poolsize
        assert (input_length - filter_length + 1) % poolsize == 0
        
        # Determine input, output, and filter tensor sizes
        conv_output_size = (input_length-filter_length+1)/poolsize
        self.input_shape = (batch_size, input_number, 1, input_length)
        self.output_shape = (batch_size, filters, 1, conv_output_size)
        self.filter_shape = (filters, input_number, 1, filter_length)
        
        # Reshape input to input size and store
        self.input = input.reshape(self.input_shape)
        
        # Model parameters (weights and biases)
        self.W = theano.shared(np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=self.filter_shape), dtype=theano.config.floatX), borrow=True)
        self.b = theano.shared(value=np.zeros(filters, dtype=theano.config.floatX), borrow=True)

        # Convolve input feature maps with filters
        conv_out = conv.conv2d(self.input, self.W, image_shape=self.input_shape, filter_shape=self.filter_shape)

        # Downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out, ds=(1, poolsize), ignore_border=True)
        
        # Store output
        self.output = activation_function(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # Store parameters of this layer
        self.params = [self.W, self.b]


class FullyConnectedLayer(object):
    """Fully connected layer of a 1-D neural network"""
    def __init__(self, rng, input, input_length, output_length, batch_size, W_bound=0.1, cost_function=sqr_error_cost):
        """Initialize fully connected layer"""
        # Determine input and output tensor sizes
        self.input_shape = (batch_size, input_length)
        self.output_shape = output_length
        
        # Store input and cost function
        self.input = input
        self.cost_function = cost_function
        
        # Model parameters (weights and biases)
        self.W = theano.shared(np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=(input_length, output_length)), dtype=theano.config.floatX), borrow=True)
        self.b = theano.shared(np.zeros((output_length,), dtype=theano.config.floatX), borrow=True)
        
        # Store output and params of this layer
        self.output = T.dot(input, self.W) + self.b
        self.params = [self.W, self.b]

    def cost(self, y):
        """Return the cost associated with the output"""
        return self.cost_function(y, self.output)
