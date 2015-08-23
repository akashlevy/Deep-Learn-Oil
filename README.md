# QRI Project at TRiCAM

<img src="logos/iacs.png" alt="IACS Logo" height="100"/>
<img src="logos/qri.png" alt="QRI Logo" height="100"/>

This repository contains the source files required to reproduce the results in "Applying Deep Learning to Petroleum Well Data." This README will explain how to use these files.

## Dependencies

- [Python 2.7](https://www.python.org/)
- [Numpy/Matplotlib](http://www.scipy.org/)
- [Theano](http://deeplearning.net/software/theano/)
- [Keras](http://keras.io/)

## Usage

### Preprocessing

In order to preprocess the data, you will need to go into the folder `datasets/` and run the script `dataset_gen.py`. This script reads in the CSV files from `data/` and converts it into chunks. It does this based on several parameters. `IN_MONTHS`, `OUT_MONTHS` and `STEP_MONTHS`, specify how many months of input, how many months of output and how often to sample for chunks. It also requires two preprocessing parameters, `REMOVE_ZEROS` and `NORMALIZE_DATA`. `REMOVE_ZEROS`, when set to true, will eliminate all zeros from the datasets and push the points together. `NORMALIZE_DATA` will normalize each chunk with respect to the input portion. The random seed `SEED` determines how the data is shuffled. As the data from each well is made into chunks, the chunks are assigned to the training, validation, and testing datasets. The wells are assigned in a train:valid:test = 6:1:1 ratio. Each dataset is represented as a tuple in Python; the first element of the tuple is a NumPy array containing the chunk inputs (the "x"), and the second element of the tuple is a NumPy array containing the chunk outputs (the "y"). The three datasets are then pickled and stored in a gzipped file called `qri.pkl.gz`. After the dataset is careated, the chunks are plotted using matplotlib.

### Testing a Single Model
In the `keras/` folder, there are several scripts with names of different neural network architectures. Each contains the code required to construct a single neural network. Each file consists of a similar structure.

#### Structure

After importing the necessary libraries, a model name is specified through `MDL_NAME`. Next, NumPy's random number generator is seeded with a number to ensure reproducibility of the neural network's results. Then the QRI data is loaded from the gzipped pickle file `qri.pkl.gz` and split into either 2D or 3D datasets. After this comes the architecture specification. The stochastic gradient descent algorithm parameters are then specified; `lr` refers to the learning rate, `momentum` specifies the extent to which past gradient values should be incorporated into the optimization, `decay` specifies the rate at which the learning rate decreases, and `nesterov` specifies whether or not Nesterov's formula should be used to compute the gradient. After the optimization technique is specified, the model is compiled with Theano using a particular loss function.

Next, the early stopping parameters are specified. The validation loss is monitored and `patience` specifies how long the neural network should wait to observe a new best validation loss. The best model is saved to the subfolder `models/<MDL_NAME>.mdl`. These features are incorporated using a callback mechanism during training.

The model is then trained. The lines
```python
t0 = time.time()
```
and
```python 
time_elapsed = time.time() - t0
```
are used to determine how long training took. There are three parameters to the training function `model.fit`; the first is `verbose` that specifies how often data should be printed to the console. The second is `nb_epoch` that specifies the maximum number of training steps. The last is `batch_size` that specifies the number of chunks that should be trained on at once.

After the model is done training, the best model is loaded from the MDL file. Then the model is evaluated on the testing set and the training time and testing set error are displayed. The results and the training/validation error are saved to `results/<MDL_NAME>.out` and `models/<MDL_NAME>.hist` respectively. Then the training and validation error are plotted as well as the test predictions.

#### Model Specifics

Every model begins with 
```python
model = Sequential()
```
which denotes that the neural network consists of a series of stacked layers. There are many different kinds of layers:
- **Dense**: a regular fully-connected layer; specify number of inputs, number of outputs, and activation function
- **Convolution1D**: a convolutional layer; specify *stack size* (how many filters you used in the previous layer, 1 if first layer), number of kernels per filter, and activation function
- **SimpleRNN, GRU, LSTM, MUT123**: different kinds of recurrent layers; specify number of inputs, number of outputs, and activation function
- **SimpleDeepRNN**: a multi-layer recurrent network; specify number of inputs, number of outputs, number of layers, and activation function
- **Dropout**: used to make a network more sparse; specify the fraction of inputs to randomly set to 0
- **Flatten**: convert a multi-dimensional input into a 1D input.

Using these Keras layers, we can construct custom neural networks to perform time series prediction on oil wells.

#### Custom Neural Network Tools (found in `qri.py`)

- `load_data`: loads the data from `qri.pkl.gz`
- `plot_test_predictions`: plots each chunk from the test set along with the prediction made for that set
- `plot_train_valid_loss`: plots how the training and validation error decreased in training
- `print_output_graph`: prints the computational graph for producing predictions to filename in a specified image format; useful for debugging and seeing how the network actually works
- `plot_weights`: plots the weight matrix for each layer in the neural network; useful for understanding what the neural network is learning
- `mae_clip`: provides a Theano expression for the mean absolute error with clipping to provide resistance to outliers; the `CLIP_VALUE` can be changed to adjust the number of standard deviations at which to begin clipping
- `save_results`: pickles the results and saves them to a file
- `save_history`: saves the training and validation loss history to a file

### Hyperparameter Optimization using Grid Search

We used variants of the scripts provided in `cluster` to run our models on Harvard's Odyssey computing cluster. They can be modified to work on different kinds of clusters.

### Bayesian Hyperparameter Optimization
For more information, see [Spearmint](https://github.com/JasperSnoek/spearmint).

## Contact
Please contact <akashl@princeton.edu> or <michelleyang@berkeley.edu> with any questions about this repository. Thank you!
