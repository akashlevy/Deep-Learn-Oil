"""Provides methods for obtaining, viewing, splitting oil production data"""

import csv
import cPickle
import gzip
import matplotlib.pyplot as plt
import numpy as np
import os
import random as rnd

# Parameters for reader
DATA_DIRECTORY = "../data"

# Splitting data
IN_MONTHS = 48
OUT_MONTHS = 12
STEP_MONTHS = 6

# Preprocessing parameters
REMOVE_ZEROS = True
NORMALIZE_DATA = True

# Random seed
SEED = 42

def get_data():
    """Returns dictionary containing data from files in data directory"""
    # Oil production data is contained in this dictionary
    # Keys are the oil well names
    # Values are lists containing oil production measurements
    data = {}
    
    # Get start directory
    startdir = os.getcwd()
    
    # Get data from files in data directory
    os.chdir(DATA_DIRECTORY)
    for filename in os.listdir(os.getcwd()):
        with open(filename, "rb") as csvfile:
            # Open each data file with csv reader
            reader = csv.reader(csvfile, dialect="excel")

            # Ignore the first line because it contains headers
            reader.next()

            # Add each row to the corresponding oil well
            for row in reader:
                # Get data from cells and convert appropriately
                well_name = row[3]
                oil = float(row[4])

                # Add data to the dictionary
                if not well_name in data:
                    data[well_name] = []
                data[well_name].append(oil)

    # Go back to start directory
    os.chdir(startdir)
    
    # Return data dictionary
    return data


def preprocess_data(data):
    """Returns preprocessed version of the data"""
    # Initialize dataset components
    train_x = []
    train_y = [] 
    valid_x = []
    valid_y = []
    test_x = []
    test_y = []
    
    # Shuffle wells
    well_names = data.keys()
    rnd.shuffle(well_names)
    
    # Go through wells and assign to datasets
    for well_index, well_name in enumerate(well_names):
        # Remove zeroed data points (push points together)
        if REMOVE_ZEROS:
            oils = np.array(filter(lambda oil: oil != 0, data[well_name]))
        else:
            oils = np.array(data[well_name])
        
        # Make chunks
        for i in xrange(0, len(oils)-(IN_MONTHS+OUT_MONTHS), STEP_MONTHS):
            # Split data into x, y, and chunk
            in_index = i
            out_index = i + IN_MONTHS
            end_index = i + IN_MONTHS + OUT_MONTHS
            chunk_x = oils[in_index:out_index]
            chunk_y = oils[out_index:end_index]
            
            # Normalize chunk w/respect to x (skip if standard deviation is 0)
            if NORMALIZE_DATA:
                mean = np.mean(chunk_x)
                std = np.std(chunk_x)
                chunk_x = (chunk_x - mean)/std
                chunk_y = (chunk_y - mean)/std
                
            # Assign to dataset based on well index
            if well_index < len(data)*6/8:
                train_x.append(chunk_x)
                train_y.append(chunk_y)
            elif well_index < len(data)*7/8:
                valid_x.append(chunk_x)
                valid_y.append(chunk_y)
            else:
                test_x.append(chunk_x)
                test_y.append(chunk_y)

    # Make datasets
    train_set = (np.array(train_x), np.array(train_y))
    valid_set = (np.array(valid_x), np.array(valid_y))
    test_set = (np.array(test_x), np.array(test_y))
    
    print "Training Set Size: %d" % train_set[0].shape[0]
    print "Validation Set Size: %d" % valid_set[0].shape[0]
    print "Test Set Size: %d" % test_set[0].shape[0]
    
    return train_set, valid_set, test_set


def plot_chunks(datasets):
    """Plots the datasets' chunks using pyplot"""
    for dataset in datasets:
        for chunk in zip(dataset[0], dataset[1]):
            # Create a figure and add a subplot with labels
            fig = plt.figure(1)
            graph = fig.add_subplot(111)
            fig.suptitle("Chunk Data", fontsize=25)
            plt.xlabel("Month", fontsize=15)
            plt.ylabel("Production", fontsize=15)
            
            # Plot the predictions as a green line with round markers
            prediction = np.append(chunk[0], chunk[1])
            graph.plot(prediction, "g-o", label="Prediction")
    
            # Plot the past as a red line with round markers
            past = chunk[0]
            graph.plot(past, "r-o", label="Past")
    
            # Add legend and display plot
            plt.legend(loc="upper left")
            plt.show()
            
            # Close the plot
            plt.close(fig)


if __name__ == '__main__':
    rnd.seed(SEED)
    print "Getting data..."
    data = get_data()
    print "Preprocessing data..."
    datasets = preprocess_data(data)
    print "Writing datasets to qri.pkl.gz..."
    with gzip.open("qri.pkl.gz", "wb") as file:
        file.write(cPickle.dumps(datasets))
    print "Done!"
    print "Plotting chunks..."
    plot_chunks(datasets)
