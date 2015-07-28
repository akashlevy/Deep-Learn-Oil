"""Test convolutional neural network"""

import nnet1d

# Create model
model = nnet1d.NNet1D(datafile="datasets/qri.pkl.gz", seed=42, batch_size=20,
                      learning_rate=0.01, momentum=0.99,
                      cost_fn=nnet1d.abs_error_cost)

# Add layers and connect them
model.add_conv_pool_layer(filters=24, filter_length=11, poolsize=2,
                          activ_fn=nnet1d.relu)
model.add_conv_pool_layer(filters=24, filter_length=12, poolsize=2,
                          activ_fn=nnet1d.relu)
model.add_fully_connected_layer()
model.build()

# Train until validation error does not improve
model.train_early_stopping()

# Save and plot data
model.save_model("cnn.mdl")
model.plot_train_valid_error()
model.plot_test_predictions()