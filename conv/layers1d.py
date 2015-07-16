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
        
        # Store poolsize and activation function
        self.poolsize = poolsize
        self.activation_function = activation_function
        
        # Reshape input to input size and store
        self.input = input.reshape(self.input_shape)
        
        # Model parameters (weights and biases)
        filts = rng.uniform(low=-W_bound, high=W_bound, size=self.filter_shape)
        dtype = theano.config.floatX
        self.W = theano.shared(np.asarray(filts, dtype=dtype), borrow=True)
        self.b = theano.shared(np.zeros(filters, dtype=dtype), borrow=True)

        # Convolve input feature maps with filters
        conv_out = conv.conv2d(self.input, self.W,
                               image_shape=self.input_shape,
                               filter_shape=self.filter_shape)

        # Downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=(1, self.poolsize),
                                            ignore_border=True)
        
        # Store output
        bias = self.b.dimshuffle('x', 0, 'x', 'x')
        self.output = self.activation_function(pooled_out + bias)

        # Store parameters of this layer
        self.params = [self.W, self.b]


class FullyConnectedLayer(object):
    """Fully connected layer of a 1-D neural network"""
    def __init__(self, rng, input, input_length, output_length, batch_size,
                 cost_function=sqr_error_cost,
                 activation_function=None, W_bound=0.1):
        """Initialize fully connected layer"""
        # Determine input and output tensor sizes
        self.input_shape = (batch_size, input_length)
        self.output_shape = output_length
        
        # Store input and cost function
        self.input = input
        self.cost_function = cost_function
        
        # Model parameters (weights and biases)'
        weight_size = (input_length, output_length)
        bias_size = (output_length,)
        weights = rng.uniform(low=-W_bound, high=W_bound, size=weight_size)
        dtype = theano.config.floatX
        self.W = theano.shared(np.asarray(weights, dtype=dtype), borrow=True)
        self.b = theano.shared(np.zeros(bias_size, dtype=dtype), borrow=True)
        
        # Store output and params of this layer
        self.output = T.dot(input, self.W) + self.b
        if activation_function is not None:
            self.output = activation_function(self.output)
        self.params = [self.W, self.b]

    def cost(self, y):
        """Return the cost associated with the output"""
        return self.cost_function(y, self.output)
