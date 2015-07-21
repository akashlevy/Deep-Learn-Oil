import numpy as np
import matplotlib.pyplot as plt

train_error = []
valid_error = []
train_error2 = []
valid_error2 = []
train_error3 = []
valid_error3 = []

with open("train_valid_data4") as f:
    for line in f.readlines():
        train_error.append(eval(line)[1])
        valid_error.append(eval(line)[2])

with open("train_valid_data2") as f:
    for line in f.readlines():
        train_error2.append(eval(line)[1])
        valid_error2.append(eval(line)[2])

with open("train_valid_data3") as f:
    for line in f.readlines():
        train_error3.append(eval(line)[1])
        valid_error3.append(eval(line)[2])

# Create a figure and add a subplot with labels
fig = plt.figure(1)
graph = fig.add_subplot(111)
fig.suptitle("Error vs. Training Steps", fontsize=25)
plt.xlabel("Epoch", fontsize=15)
plt.ylabel("Absolute Error", fontsize=15)

# Plot the training error as a green line with round markers
graph.plot(train_error, label="Training Set 1")

# Plot the validation error as a red line with round markers
graph.plot(valid_error, label="Validation Set 1")

# Plot the training error as a green line with round markers
#graph.plot(train_error2, label="Training Set 2")

# Plot the validation error as a red line with round markers
#graph.plot(valid_error2, label="Validation Set 2")

# Plot the training error as a green line with round markers
#graph.plot(train_error3, label="Training Set 3")

# Plot the validation error as a red line with round markers
#graph.plot(valid_error3, label="Validation Set 3")

# Add legend and display plot
plt.legend()
plt.show()