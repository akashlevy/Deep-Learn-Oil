"""Test nnet1d"""

from nnet1d import NNet1D
from nnet_functions import abs_error_cost, tanh

# Fully connected model 1
model = NNet1D(datafile="../datasets/qri.pkl.gz", seed=42, batch_size=200, learning_rate=0.01, momentum=0, cost_fn=abs_error_cost)
# model.add_conv_pool_layer(filters=24, filter_length=11, poolsize=2, activ_fn=tanh)
# model.add_conv_pool_layer(filters=24, filter_length=12, poolsize=2, activ_fn=tanh)
model.add_fully_connected_layer()
model.build()
model.train_early_stopping(patience=15, min_epochs=100)
# model.plot_test_predictions(display_figs=False, save_figs=True, output_folder="conv_images")