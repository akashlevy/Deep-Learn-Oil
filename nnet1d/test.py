"""Test nnet1d"""

from nnet1d import NNet1D
from nnet_fns import abs_error_cost, tanh, relu

# Fully connected model 1
model = NNet1D(datafile="../datasets/qri.pkl.gz", seed=42, batch_size=200, learning_rate=0.2, momentum=0.99, cost_fn=abs_error_cost)
# model.add_conv_pool_layer(filters=15, filter_length=11, poolsize=2, activ_fn=tanh)
# model.add_conv_pool_layer(filters=15, filter_length=12, poolsize=2, activ_fn=tanh)
model.add_fully_connected_layer(output_length=30, activ_fn=relu)
model.add_fully_connected_layer(output_length=30, activ_fn=relu)
model.add_fully_connected_layer(output_length=30, activ_fn=relu)
model.add_fully_connected_layer(output_length=30, activ_fn=relu)
model.add_fully_connected_layer()
model.build()
model.train_early_stopping(patience=5)
model.save_model("last_model.mdl")
model.plot_train_valid_error()
model.plot_test_predictions()