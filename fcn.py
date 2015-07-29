"""Test fully connected neural network"""

from nnet1d import NNet1D, relu, abs_error_cost

# Create model
model = NNet1D(datafile="datasets/qri.pkl.gz", seed=42, batch_size=20,
               learning_rate=0.01, momentum=0.99, cost_fn=abs_error_cost)

# Add layers and connect them
model.add_fully_connected_layer(output_length=30, activ_fn=relu)
model.add_fully_connected_layer(output_length=30, activ_fn=relu)
model.add_fully_connected_layer()
model.build()
model.print_output_graph("out")

# Train until validation error does not improve
model.train_early_stopping()

# Save and plot data
model.save_model("models/fcn.mdl")
model.plot_train_valid_error()
model.plot_test_predictions()