"""Layer classes of a 1-D neural network"""

import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from nnet_functions import relu, abs_error_cost


# Configure floating points for Theano
theano.config.floatX = "float32"
dtype = theano.config.floatX


class Layer(object):
    """Layer of a 1-D neural network"""
    def __init__(self, input, input_length, activ_fn):
        """Store required layer attributes"""
        self.input = input
        self.input_length = input_length
        self.activ_fn = activ_fn
    
    @staticmethod
    def shared_uniform(rng, size, W_bound=0.01):
        """Initialize a matrix shared variable with uniformly distributed
        elements"""
        weights = rng.uniform(low=-W_bound, high=W_bound, size=size)
        return theano.shared(np.asarray(weights, dtype=dtype), borrow=True)

    @staticmethod
    def shared_zeros(*shape):
        """Initialize a vector shared variable with zero elements."""
        return theano.shared(np.zeros(shape, dtype=dtype), borrow=True)


class ConvPoolLayer(Layer):
    """Convolutional layer of a 1-D neural network"""
    def __init__(self, rng, input, input_length, filters, filter_length,
                 input_number=1, poolsize=1, activ_fn=relu, W_bound=0.01):
        """Initialize layer"""
        # Make sure that convolution output is evenly divisible by poolsize
        assert (input_length - filter_length + 1) % poolsize == 0
        
        # Determine input and output length as well as filter shape
        self.output_length = (input_length - filter_length + 1) / poolsize
        self.filter_shape = (filters, input_number, 1, filter_length)
        
        # Store input, input length, activation function, and poolsize
        Layer.__init__(self, input, input_length, activ_fn)
        self.poolsize = poolsize
        
        # Model parameters (weights and biases)
        self.W = Layer.shared_uniform(rng, self.filter_shape, W_bound)
        self.b = Layer.shared_zeros(filters)

        # Convolve input feature maps with filters
        conv_out = conv.conv2d(self.input, self.W, None, self.filter_shape)

        # Downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out, ds=(1, poolsize))
        
        # Store output
        self.output = activ_fn(pooled_out + self.b.dimshuffle('x',0,'x','x'))

        # Store parameters of this layer
        self.params = [self.W, self.b]

    def __repr__(self):
        """Return string representation of ConvPoolLayer"""
        format_line = "ConvPoolLayer(rng, input, %s, %s, %s, "
        format_line += "input_number=%s, poolsize=%s, activ_fn=%s)"
        activ_fn_name = self.activ_fn.__name__ if self.activ_fn else "None"
        return format_line % (self.input_length, self.filter_shape[0],
                              self.filter_shape[3], self.filter_shape[1],
                              self.poolsize, activ_fn_name)
    
    def __str__(self):
        """Return string representation of ConvPoolLayer"""
        return self.__repr__()

    def plot_filters(self, cmap="gray"):
        """Plot the filters"""
        filters = self.W.get_value(borrow=True)
        new_shape = (-1, filters.shape[3])
        filters = np.resize(filters, new_shape)
        fig = plt.figure(1)
        graph = fig.add_subplot(111)
        mat = graph.matshow(filters, cmap=cmap, interpolation="none")
        fig.colorbar(mat)
        plt.show()


class FullyConnectedLayer(Layer):
    """Fully connected layer of a 1-D neural network"""
    def __init__(self, rng, input, input_length, output_length, activ_fn=None,
                 cost_fn=abs_error_cost, W_bound=0.01):
        """Initialize fully connected layer"""
        # Store layer parameters, cost function, output length
        Layer.__init__(self, input, input_length, activ_fn)
        self.cost_fn = cost_fn
        self.output_length = output_length
        
        # Model parameters (weights and biases)'
        self.W = Layer.shared_uniform(rng, (input_length, output_length),
                                      W_bound)
        self.b = Layer.shared_zeros(output_length)
        
        # Store output and params of this layer
        self.output = T.dot(input, self.W) + self.b
        if activ_fn is not None:
            self.output = self.activ_fn(self.output)
        self.params = [self.W, self.b]

    def __repr__(self):
        """Return string representation of FullyConnectedLayer"""
        format_line = "FullyConnectedLayer(rng, input, %s, %s, activ_fn=%s, "
        format_line += "cost_fn=%s)"
        activ_fn_name = self.activ_fn.__name__ if self.activ_fn else "None"
        cost_fn_name = self.cost_fn.__name__ if self.cost_fn else "None"
        return format_line % (self.input_length, self.output_length,
                              activ_fn_name, cost_fn_name)
    
    def __str__(self):
        """Return string representation of FullyConnectedLayer"""
        return self.__repr__()
    
    def cost(self, y):
        """Return the cost associated with the output"""
        return self.cost_fn(y, self.output)
        
    def plot_weights(self, cmap="gray"):
        """Plot the weight matrix"""
        weights = self.W.get_value(borrow=True)
        fig = plt.figure(1)
        graph = fig.add_subplot(111)
        mat = graph.matshow(weights, cmap=cmap, interpolation="none")
        fig.colorbar(mat)
        plt.show()


class RecurrentLayer(FullyConnectedLayer):
    """Recurrent layer of neural network"""
    def __init__(self, rng, input, input_length, output_length, activ_fn=relu,
                 W_bound=0.01):
        """Initialize recurrent layer"""
        # Treat as fully connected layer
        FullyConnectedLayer.__init__(self, rng, input, input_length,
                                     output_length, activ_fn)

        # Recurrent weights and stored hidden state
        self.W_r = Layer.shared_uniform(rng, (output_length, output_length),
                                        W_bound)
        self.h = Layer.shared_zeros(output_length)

        # Store params of this layer
        self.params = [self.W, self.W_r, self.b]
        
        # Recurrent step function
        def step(x):
            self.h = T.dot(self.h, self.W_r) + T.dot(x, self.W) + self.b
            self.h = self.activ_fn(self.h)
            return self.h
        
        # Compute hidden layer as output
        self.output, _ = theano.scan(step, self.input)
    
    def __repr__(self):
        """Return string representation of RecurrentLayer"""
        format_line = "RecurrentLayer(rng, input, %s, %s, activ_fn=%s, "
        format_line += "cost_fn=%s)"
        activ_fn_name = self.activ_fn.__name__ if self.activ_fn else "None"
        return format_line % (self.input_length, self.output_length,
                              activ_fn_name)
    
    def __str__(self):
        """Return string representation of RecurrentLayer"""
        return self.__repr__()
        
    def plot_recurrent_weights(self, cmap="gray"):
        """Plot the recurrent weight matrix"""
        weights = self.W_r.get_value(borrow=True)
        fig = plt.figure(1)
        graph = fig.add_subplot(111)
        mat = graph.matshow(weights, cmap=cmap, interpolation="none")
        fig.colorbar(mat)
        plt.show()
