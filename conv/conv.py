"""ADD IN MOMENTUM?"""

import theano.tensor as T
from theano import shared, function
from convolutional_mlp import LeNetConvPoolLayer
from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer

# Theano settings and random number generator
theano.config.floatX = "float32"
rng = np.random.RandomState(42)

# Load data sets
dataset = "mnist.pkl.gz"
#dataset = "../qri.pkl.gz"
datasets = load_data(dataset)
train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]
print "Size of the training data matrix: ", train_set_x.get_value().shape

# Network parameters
input_size = (6, 6)
num_kernels = [9, 9]
kernel_sizes = [(3, 3), (2, 2)]
output_size = 12
pool_size = (1, 1)

# Training parameters
learning_rate = 0.1
batch_size = 1

# Input and output
x = T.matrix('x')
y = T.vector('y')

# Split into batches
n_train_batches = train_set_x.get_value(borrow=True).shape[0]
n_test_batches = test_set_x.get_value(borrow=True).shape[0]
n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
n_train_batches /= batch_size
n_test_batches /= batch_size
n_valid_batches /= batch_size

# Check that we have a multiple of pool_size before pooling
assert ((28 - kernel_sizes[0][0] + 1) % pool_size[0]) == 0

# Get input and output sizes
layer0_input_size = (batch_size, 1, input_size[0], input_size[1])
edge0 = (6 - kernel_sizes[0][0] + 1)/pool_size[0]
layer0_output_size = (batch_size, num_kernels[0], edge0, edge0)

# Reshape placeholder x to the input size of the network
layer0_input = x.reshape(layer0_input_size)

filter_shape = (num_kernels[0], 1) + kernel_sizes[0]
layer0 = LeNetConvPoolLayer(rng, layer0_input, layer0_input_size, filter_shape,
                            pool_size)


layer0 = LeNetConvPoolLayer(rng, layer0_input, layer0_input_size, (num_kernels[0], 1) + kernel_sizes[0], (1, 1))