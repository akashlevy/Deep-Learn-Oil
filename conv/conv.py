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
batch_size = 50

layer0 = conv1d.ConvPoolLayer(rng, x, input_length=36, batch_size=batch_size, filters=10, filter_length=5, poolsize=4)
layer1 = conv1d.ConvPoolLayer(rng, layer0.output, input_number=layer0.output_shape[1], input_length=layer0.output_shape[3], batch_size=batch_size, filters=10, filter_length=4, poolsize=5)
layer2 = conv1d.FullyConnectedLayer(rng, layer1.output.flatten(2), n_in=layer1.output_shape[1]*layer1.output_shape[3], n_out=12)
network = conv1d.ConvNetwork(datasets, x, y, batch_size=batch_size, layers=[layer0, layer1, layer2], learning_rate=0.1)

# Train and test neural network
network.train(100)
network.test()