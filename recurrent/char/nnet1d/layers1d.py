"""Layer classes of a 1-D neural network"""

import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from nnet_fns import relu, abs_error_cost


class Layer(object):
    """Layer of a 1-D neural network"""
    def __init__(self, input, input_length, activ_fn):
        """Store required layer attributes"""
        self.input = input
        self.input_length = input_length
        self.activ_fn = activ_fn


class FullyConnectedLayer(Layer):
    """Fully connected layer of a 1-D neural network"""
    def __init__(self, rng, input, input_length, output_length, activ_fn=None,
                 output_type='real', cost_fn=abs_error_cost, W_bound=0.01):
        """Initialize fully connected layer"""
        # Store layer parameters, cost function, output length
        super(FullyConnectedLayer,self).__init__(input, input_length, activ_fn)
        self.cost_fn = cost_fn
        self.output_length = output_length
        
        # Model parameters (weights and biases)'
        weight_size = (input_length, output_length)
        bias_size = (output_length,)
        weights = rng.uniform(low=-W_bound, high=W_bound, size=weight_size)

        if output_type == 'real':
            dtype = theano.config.floatX
        elif output_type == 'softmax':
            dtype = 'int32'
        else:
            raise NotImplementedError

        self.W = theano.shared(np.asarray(weights, dtype=theano.config.floatX), borrow=True)
        self.b = theano.shared(np.zeros(bias_size, dtype=theano.config.floatX), borrow=True)
        
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


class RecurrentLayer(Layer):
    """Recurrent layer of neural network"""
    def __init__(self, rng, input, input_length, output_length, activ_fn=relu,
                 output_type='real', W_bound=0.01):
        """Initialize recurrent layer"""
        # Store layer parameters and output length
        super(RecurrentLayer, self).__init__(input, input_length, activ_fn)
        self.output_length = output_length

        # Recurrent weights
        weight_size = (output_length, output_length)
        weights = rng.uniform(low=-W_bound, high=W_bound, size=weight_size)
        
        if output_type == 'real':
            dtype = theano.config.floatX
        elif output_type == 'softmax':
            dtype = 'int32'
        else:
            raise NotImplementedError

        self.W = theano.shared(value=np.asarray(weights, dtype=theano.config.floatX))
        
        # Input to hidden layer weights
        weight_size = (input_length, output_length)
        weights = rng.uniform(low=-W_bound, high=W_bound, size=weight_size)
        self.W_in = theano.shared(value=np.asarray(weights, dtype=dtype))

        # Initial state of hidden layer
        self.h0 = theano.shared(np.zeros((output_length,), dtype=dtype))

        # Bias of hidden layer
        self.b = theano.shared(np.zeros((output_length,), dtype=dtype))

        # Store output and params of this layer
        self.params = [self.W, self.W_in, self.h0, self.b]

        # Recurrent function
        def step(x_t, h_tm1):
            self.h_t = T.dot(x_t, self.W_in) + T.dot(h_tm1, self.W) + self.b
            self.h_t = self.activ_fn(self.h_t)
            return self.h_t

        # Get output
        self.output, _ = theano.scan(step, self.input, outputs_info=self.h0)
    
    def __repr__(self):
        """Return string representation of FullyConnectedLayer"""
        format_line = "RNNLayer(rng, input, %s, %s, activ_fn=%s, cost_fn=%s)"
        activ_fn_name = self.activ_fn.__name__ if self.activ_fn else "None"
        return format_line % (self.input_length, self.output_length,
                              activ_fn_name)
    
    def __str__(self):
        """Return string representation of FullyConnectedLayer"""
        return self.__repr__()
        
    def plot_weights(self, cmap="gray"):
        """Plot the weight matrix"""
        weights = self.W_in.get_value(borrow=True)
        fig = plt.figure(1)
        graph = fig.add_subplot(111)
        mat = graph.matshow(weights, cmap=cmap, interpolation="none")
        fig.colorbar(mat)
        plt.show()
        
    def plot_recurrent_weights(self, cmap="gray"):
        """Plot the rweight matrix"""
        weights = self.W.get_value(borrow=True)
        fig = plt.figure(1)
        graph = fig.add_subplot(111)
        mat = graph.matshow(weights, cmap=cmap, interpolation="none")
        fig.colorbar(mat)
        plt.show()
