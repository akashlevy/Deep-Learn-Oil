""" Miscellaneous functions """
import theano.tensor as T

def sqr_error_cost(y, output):
    """ Return the average square error between output vector and y in Theano """
    return T.mean(T.sqr(y - output))

def abs_error_cost(y, output):
    """ Return the average absolute error between output vector and y in
    Theano """
    return T.mean(T.abs_(y - output))

def std_abs_error(y, output):
    return T.std(T.abs_(y - output))