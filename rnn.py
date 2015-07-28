"""Test recurrent neural network"""

import nnet1d

# Create model
<<<<<<< HEAD
model = nnet1d.NNet1D(datafile="datasets/qri.pkl.gz", seed=42, batch_size=20,
=======
model = nnet1d.nnet1d.NNet1D(datafile="datasets/qri.pkl.gz", seed=42, batch_size=50,
>>>>>>> 2f07e0e1d0d18fef445ce1a9f90388cc0cfc103b
                      learning_rate=0.01, momentum=0.99,
                      cost_fn=nnet1d.abs_error_cost)

# Add layers and connect them
model.add_recurrent_layer(output_length=500, activ_fn=nnet1d.relu)
model.add_fully_connected_layer()
model.build()

# Train until validation error does not improve
model.train_early_stopping()

# Save and plot data
# model.save_model("rnn.mdl")
model.plot_train_valid_error()
model.plot_test_predictions()