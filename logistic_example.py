import numpy
import theano
import theano.tensor as T
from matplotlib import pyplot as plt
rng = numpy.random

# Parameters
N = 400
feats = 784
D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
training_steps = 500

# Declare Theano symbolic variables
x = T.matrix("x")
y = T.vector("y")
w = theano.shared(rng.randn(feats), name="w")
b = theano.shared(0., name="b")

# Initial model
print "Initial model:"
print w.get_value(), b.get_value()
print

# Construct Theano expression graph
p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))   # Probability that target = 1
prediction = p_1 > 0.5                    # The prediction thresholded
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # Cross-entropy loss function
cost = xent.mean() + 0.01 * (w ** 2).sum()# The cost to minimize
gw, gb = T.grad(cost, [w, b])             # Compute the gradient of the cost
                                          # (we shall return to this in a
                                          # following section of this tutorial)

# Compile
train = theano.function(
          inputs=[x,y],
          outputs=[prediction, xent],
          updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))

# Prediction function
predict = theano.function(inputs=[x], outputs=prediction)

# Train
for i in range(training_steps):
    pred, err = train(D[0], D[1])

# Final model output
print "Final model:"
print w.get_value(), b.get_value()
print
print "target values for D:"
print D[1]
print
print "prediction on D:"
print predict(D[0])

# Plot to see similarities and differences
fig = plt.figure(1)
graph = fig.add_subplot(111)
graph.plot(range(400), D[1], "r-o", label="Target")
graph.plot(range(400), predict(D[0]), "g-o", label="Prediction")
plt.legend()
plt.show()