import numpy as np
import matplotlib.pyplot as plt
import os

batch_sizes = []
test_error = []

# Get data from files in data directory
for filename in os.listdir(os.getcwd()):
    with open(filename) as f:
        if filename[-4:] == ".out":
            batch_sizes.append(int(filename[9:-4]))
            f.readline()
            test_error.append(float(f.readline()))

# Sort data points
batch_sizes, test_error = zip(*sorted(zip(batch_sizes, test_error)))

# Create a figure and add a subplot with labels
fig = plt.figure(1)
graph = fig.add_subplot(111)
fig.suptitle("Testing Error vs. Batch Size", fontsize=25)
graph.set_xlabel("Batch Size", fontsize=15)
graph.set_ylabel("Testing Error", fontsize=15)

# Plot the testing error as a red line
graph.plot(batch_sizes, test_error, 'r', label="Testing Error")

# Add legend and display plot
plt.legend()
plt.show()