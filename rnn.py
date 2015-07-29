"""Test recurrent neural network"""

import nnet1d

# Create model
model = nnet1d.NNet1D(datafile="datasets/qri.pkl.gz", seed=42, batch_size=20,
                      learning_rate=0.01, momentum=0.99,
                      cost_fn=nnet1d.abs_error_cost)

# Add layers and connect them
model.add_recurrent_layer(output_length=30, activ_fn=nnet1d.relu)
model.add_fully_connected_layer()
model.build()
model.print_output_graph("out")

# Train until validation error does not improve
model.train_early_stopping()

# Plot matrices
model.layers[0].plot_recurrent_weights(cmap=None)
model.layers[0].plot_weights(cmap=None)

# Save and plot data
model.save_model("models/rnn.mdl")
model.plot_train_valid_error()
model.plot_test_predictions()