#!/usr/bin/env python

"""interp.py: performs polynomial interpolaton on QRI data in .csv format"""

__author__ = "Ruomeng (Michelle) Yang"
__credits__ = ["Akash Levy"]

import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy

from datetime import datetime
from matplotlib.dates import date2num, num2date
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Function to approximate by polynomial interpolation
def approx(x):
    return x * np.sin(x)

# Data is contained in this dictionary
# Keys are the oil well names
# Values are lists containing two lists that contain the dates/oil measurements
data = {}

# Data files are located in data directory
os.chdir("data")

for filename in os.listdir(os.getcwd()):
    with open(filename, "rb") as csvfile:
        # Open each data file with csv reader
        reader = csv.reader(csvfile, dialect="excel")
        
        # Ignore the first line because it contains headers
        reader.next()
        
        # Add each row to the current oil well
        for row in reader:
            # Get data from cells and convert appropriately
            date = date2num(datetime.strptime(row[0],"%m/%d/%Y"))
            name = row[3]
            oil = float(row[4])
            
            # Add data to the dictionary
            if not name in data:
                data[name] = [[], []]
            data[name][0].append(date)
            data[name][1].append(oil)

for name in data:
    dates = data[name][0]
    oils = data[name][1]

    # Create a figure and add a subplot with labels
    fig = plt.figure(1)
    graph = fig.add_subplot(111)
    fig.suptitle(name)
    plt.xlabel('Date')
    plt.ylabel('Production')
    
    # Set the xtick locations to correspond to just the dates you entered.
    graph.set_xticks(data[name][0][0::12])
     
    # Set the xtick labels to correspond to just the dates you entered.
    graph.set_xticklabels([num2date(date).strftime("%m/%y") for date in data[name][0][0::12]])
    
    # Remove zeroed data points
    i = 0
    while i < len(oils):
        if oils[i] == 0:
            del dates[i]
            del oils[i]
        else:
            i += 1
    
    # x = np.asarray(oils)
    # y = approx(x)
    # plt.plot(x, y, label="ground truth")

    # x_plot = np.asarray(oils)
    # rng = np.random.RandomState(0)
    # rng.shuffle(x_plot)
    # x_plot = np.sort(x_plot[:20])
    # y = approx(x_plot)

    # X = x_plot[:, np.newaxis]
    # X_plot = x[:, np.newaxis]

    # plt.scatter(x_plot, y, label="training points")

    # model = make_pipeline(PolynomialFeatures(5), Ridge())
    # model.fit(X, y)
    # y_plot = model.predict(X_plot)
    # plt.plot(x, y_plot, label="degree 5")
    
    # Plot the data as a red line with round markers
    graph.plot(dates, oils, "r-o")

    plt.legend(loc="lower left")
    plt.show()
