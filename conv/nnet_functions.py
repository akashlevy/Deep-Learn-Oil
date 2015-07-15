"""Miscellaneous activation functions and cost functions"""

import theano.tensor as T


def relu(x):
    """Rectified linear units activation function implemented using Theano"""
    return T.switch(x < 0, 0, x)


def sqr_error_cost(y, output):
    """Return the average square error between output vector and y in Theano"""
    return T.mean(T.sqr(y - output))