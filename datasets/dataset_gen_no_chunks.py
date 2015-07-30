"""Provides methods for obtaining, viewing, splitting oil production data"""

import csv
import cPickle
import gzip
import matplotlib.pyplot as plt
import numpy as np
import os
import random as rnd

# Splitting data
OUT_MONTHS = 12

# Parameters for reader
DATA_DIRECTORY = "../data"

# Preprocessing parameters
REMOVE_ZEROS = True
SMOOTH_DATA = False
NORMALIZE_DATA = True
SMOOTH_LEN = 4

# Random seed
SEED = 42

# Dataset assignment
DIFFERENT_WELLS = True
DIFFERENT_SITES = False
TRAIN_SITES = ["BEAP", "BEAT", "BEZE", "EUZE", "EUAP"]
TEST_SITES = ["EUAT"]

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
            
        # Smooth data
        if SMOOTH_DATA:
            smooth_window = np.ones(SMOOTH_LEN)/SMOOTH_LEN
            oils = np.convolve(smooth_window, oils, mode="valid")
            
        # Get x and y (if enough data in well)
        if len(oils) > OUT_MONTHS:
            x = oils[:-12]
            y = oils[-12:]
            if np.std(x) == 0:
                continue
        else:
            continue
            
        # Normalize chunk w/respect to x (skip if standard deviation is 0)
        if NORMALIZE_DATA:
            mean = np.mean(x)
            std = np.std(x)
            x = (x - mean)/std
            y = (y - mean)/std
        
        # Add chunk
        if DIFFERENT_SITES:
            # Assign to dataset based on site name
            if well_name[:4] in TRAIN_SITES:
                train_x.append(x)
                train_y.append(y)
            elif well_name[:4] in TEST_SITES:
                test_x.append(x)
                test_y.append(y)
            else:
                print "Error: site %s not classified" % name
        elif DIFFERENT_WELLS:
            # Assign to dataset based on well index
            if well_index < len(data)*7/8:
                train_x.append(x)
                train_y.append(y)
            else:
                test_x.append(x)
                test_y.append(y)
        else:
            print "Error: choose a dataset assignment option"

    # Make datasets
    train_set = train_x, train_y
    test_set = test_x, test_y
    
    print "Training Set Size: %d" % len(train_set[0])
    print "Test Set Size: %d" % len(test_set[0])
    
    return train_set, test_set


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
    print "Writing datasets to qri_no_chunks.pkl.gz..."
    with gzip.open("qri_no_chunks.pkl.gz", "wb") as file:
        file.write(cPickle.dumps(datasets))
    print "Done!"
    print "Plotting chunks..."
    plot_chunks(datasets)
