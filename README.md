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

In order to preprocess the data, you will need to go into the folder `datasets/` and run the script `dataset_gen.py`. This script reads in the CSV files from `data/` and converts it into chunks. It does this based on several parameters. `IN_MONTHS`, `OUT_MONTHS` and `STEP_MONTHS`, specify how many months of input, how many months of output and how often to sample for chunks. It also requires two preprocessing parameters, `REMOVE_ZEROS` and `NORMALIZE_DATA`. `REMOVE_ZEROS`, when set to true, will eliminate all zeros from the datasets and push the points together. `NORMALIZE_DATA` will normalize each chunk with respect to the input portion. The random seed `SEED` determines how the data is shuffled. As the data from each well is made into chunks, the chunks are assigned to the training, validation, and testing datasets. The wells are assigned in a train:valid:test = 6:1:1 ratio. Each dataset is represented as a tuple in Python; the first element of the tuple is a numpy array containing the chunk inputs (the "x"), and the second element of the tuple is a NumPy array containing the chunk outputs (the "y"). The three datasets are then stored in a gzipped file called `qri.pkl.gz`. After the dataset is created, the chunks are plotted using matplotlib.

#### Testing a Single Model
In the keras/ folder, there are several files.

#### Other folders
cluster: contains basic scripts for running experiments on the Odyssey cluster with and without the GPU
#### datasets/
##### DATA_Harvard_2012_10_31.xls
The raw data file we were given
##### dataset_gen.py
Reads and preprocesses the data from the data folder, generating a file called qri.pkl.gz. Parameters that can be specified are 
