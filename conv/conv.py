import conv1d
import numpy as np
import theano.tensor as T

# Initialize random number generator
rng = np.random.RandomState(42)

# Load data sets
dataset_file = "../datasets/qri.pkl.gz"
datasets = conv1d.load_data(dataset_file)

# Input and output
x = T.matrix('x')
y = T.matrix('y')

# Make neural network
layer0 = conv1d.ConvPoolLayer(rng, x, input_length=36, batch_size=25, filters=10, filter_length=5, poolsize=4)
print layer0.input_size, layer0.output_size, layer0.filter_size
layer1 = conv1d.ConvPoolLayer(rng, layer0.output, input_number=10, input_length=8, batch_size=25, filters=10, filter_length=4, poolsize=5)
print layer1.input_size, layer1.output_size, layer1.filter_size
layer2 = conv1d.FullyConnectedLayer(rng, layer1.output.flatten(), n_in=100, n_out=12)
network = conv1d.ConvNetwork(datasets, x, y, batch_size=25, layers=[layer0, layer1, layer2], learning_rate=0.1)

# Train and test neural network
network.train(10)
network.test()