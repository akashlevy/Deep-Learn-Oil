import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os


nlayers = []
nneurons = []
time_elapsed = []
test_error = []

# Get data from files in data directory
for filename in os.listdir(os.getcwd()):
    with open(filename) as f:
        if filename[-4:] == ".out":
            nlayers.append(int(filename[6]))
            nneurons.append(int(filename[19:-4]))
            time_elapsed.append(float(f.readline()))
            test_error.append(float(f.readline()))

# Create a figure and add a subplot with labels
fig = plt.figure()
graph = fig.gca(projection="3d")
fig.suptitle("Testing Error vs. HLs and Neurons/HL", fontsize=25)
graph.set_xlabel("Neurons/HL")
graph.set_ylabel("Number of Hidden Layers")
graph.set_zlabel("Testing Error")

# Plot the training error
graph.plot_trisurf(nneurons, nlayers, test_error, linewidth=0, cmap=cm.coolwarm)

# Display plot
plt.show()