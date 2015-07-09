"""Provides methods for obtaining, viewing, splitting oil production data"""
import csv
import cPickle
import gzip
import matplotlib.pyplot as plt
import numpy as np
import os
import random as rnd

SEED = 42
SETS = 500
SAMPLES = 100
NOISE = 0.01

def plot_curves(data):
	"""Plots the chunks using pyplot"""
	for curve in data[0]:
		# Create a figure and add a subplot with labels
		fig = plt.figure(1)
		graph = fig.add_subplot(111)
		fig.suptitle("Data", fontsize=25)
		plt.xlabel("X", fontsize=15)
		plt.ylabel("Y", fontsize=15)
		
		# Plot the data as a green line with round markers
		graph.plot(curve, "g-o", label="Data")

		# Add legend, resize windows, and display plot
		plt.legend()
		mng = plt.get_current_fig_manager()
		mng.resize(*mng.window.maxsize())
		plt.show()


def split_data_sets(data):
	"""Generate the training, validation, testing sets by splitting the data
	using 6:1:1 ratio"""
	train_set = (data[0][:6*len(data)/8],
				 data[1][:6*len(data)/8])
	valid_set = (data[0][6*len(data)/8:7*len(data)/8],
				 data[1][6*len(data)/8:7*len(data)/8])
	test_set = (data[0][7*len(data)/8:],
			    data[1][7*len(data)/8:])
	
	return train_set, valid_set, test_set


def generate_data():
	"""Generate the data by sampling along a normal curve and adding error"""
	data_x = []
	data_y = []
	sample_points = np.linspace(-3, 3, SAMPLES)
	# numpy.random.seed(SEED)
	for set in xrange(SETS):
		# noise = np.random.normal(0, NOISE, SAMPLES)
		curve = np.exp(-sample_points*sample_points/2)/(2*np.pi)
		data_x.append(curve.tolist())
		data_y.append(curve.tolist())
	return np.array(data_x), np.array(data_y)

if __name__ == '__main__':
	rnd.seed(SEED)
	print "Generating data..."
	data = generate_data()
	print "Splitting into data sets..."
	data_sets = split_data_sets(data)
	print "Writing data sets to da_sanity.pkl.gz..."
	print data_sets[0]
	with gzip.open("da_sanity.pkl.gz", "wb") as file:
		file.write(cPickle.dumps(data_sets))
	print "Done!"
	print "Plotting curves..."
	plot_curves(data)
