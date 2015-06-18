import csv
import numpy as np
import os
from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib.dates import date2num, num2date
from smooth import smooth

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
	# Get data from dictionary
	dates = data[name][0]
	oils = data[name][1]
	
	# Create a figure and add a subplot with labels
	fig = plt.figure(1)
	graph = fig.add_subplot(111)
	fig.suptitle(name, fontsize=25)
	plt.xlabel("Date", fontsize=15)
	plt.ylabel("Production", fontsize=15)
	
	# Set the xtick locations to correspond to the dates every 12 months
	graph.set_xticks(dates[0::12])
	 
	# Set the xtick labels to correspond to the dates every 12 months
	date_labels = [num2date(date).strftime("%m/%y") for date in dates[0::12]]
	graph.set_xticklabels(date_labels)
	
	# Remove zeroed data points
	i = 0
	while i < len(oils):
		if oils[i] == 0:
			del dates[i]
			del oils[i]
		else:
			i += 1
			
	# Convert data to numpy arrays
	dates = np.array(dates)
	oils = np.array(oils)
	
	# Plot the raw data as a red line with round markers
	graph.plot(dates, oils, "r-o", label="Oil Production")
	
	# Performs convolutional smoothing (BAD: ASSUMES POINTS ARE SPACED EVENLY)
	WINDOW_LEN = 6
	if len(oils) > WINDOW_LEN:
		s = np.r_[oils[WINDOW_LEN/2:0:-1], oils, oils[-1:-WINDOW_LEN/2:-1]]
		oils = np.convolve(np.ones(WINDOW_LEN)/WINDOW_LEN, s, mode="valid")
 	
	# Plot the smoothed data as a green line with round markers
	graph.plot(dates, oils, "g-o", label="Smoothed Oil Production")
	plt.legend()
	mng = plt.get_current_fig_manager()
	mng.resize(*mng.window.maxsize())
	plt.show()
