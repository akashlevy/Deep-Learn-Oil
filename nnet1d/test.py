"""Test nnet1d"""

from nnet1d import NNet1D
from nnet_fns import abs_error_cost, tanh

# Fully connected model 1
model = NNet1D(datafile="../datasets/qri.pkl.gz", seed=42, batch_size=200, learning_rate=0.01, momentum=0, cost_fn=abs_error_cost)
model.add_conv_pool_layer(filters=15, filter_length=5, poolsize=4, activ_fn=tanh)
model.add_conv_pool_layer(filters=15, filter_length=4, poolsize=5, activ_fn=tanh)
# model.add_fully_connected_layer(output_length=30, activ_fn=tanh)
# model.add_fully_connected_layer(output_length=24, activ_fn=tanh)
# model.add_fully_connected_layer(output_length=18, activ_fn=tanh)
# model.add_fully_connected_layer(output_length=12, activ_fn=tanh)
model.add_fully_connected_layer()
model.build()
model.train_early_stopping(min_epochs=15, max_epochs=15, patience=15)
# model.save_model("last_model.mdl")
# model.plot_train_valid_error()
# model.plot_test_predictions()
model.layers[0].plot_filters(cmap=None)
