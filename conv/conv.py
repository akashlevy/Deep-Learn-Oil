"""ADD IN MOMENTUM?"""

import theano.tensor as T
import time
from theano import shared, function

# Add path for pre-written deep learning code
sys.path.insert(1, "ComputeFest2015_DeepLearning/DeepLearningTutorials/code")
from convolutional_mlp import LeNetConvPoolLayer
from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer

# Theano settings and random number generator
theano.config.floatX = "float32"
rng = np.random.RandomState(42)

# Load data sets
dataset = "mnist.pkl.gz"
datasets = load_data(dataset)
train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]
print "Size of the training data matrix: ", train_set_x.get_value().shape

input_size = 36
output_size = 12

# Network parameters
num_kernels = [72, 48]
kernel_sizes = [12, 6]
sigmoidal_output_size = 12

# Training parameters
learning_rate = 0.05

# Input and output
x = T.vector('x')
y = T.vector('y')

layer0 = LeNetConvPoolLayer(rng, layer0_input, layer0_input_size, (num_kernels[0], 1) + kernel_sizes[0], (1, 1))