import numpy
import theano
import theano.tensor as T
from matplotlib import pyplot as plt
rng = numpy.random

N = 400
feats = 784
D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
training_steps = 10000

# Declare Theano symbolic variables
x = T.matrix("x")
y = T.vector("y")
w = theano.shared(rng.randn(feats), name="w")
b = theano.shared(0., name="b")
print "Initial model:"
print w.get_value(), b.get_value()
print
fig = plt.figure(1)
graph = fig.add_subplot(111)
graph.plot(range(T.shape(w)), w)
graph.plot(range(T.shape(w)+1), T.zeros_like(w).append(b))
fig.show()

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
predict = theano.function(inputs=[x], outputs=prediction)

# Train
for i in range(training_steps):
    pred, err = train(D[0], D[1])

print "Final model:"
print w.get_value(), b.get_value()
print
print "target values for D:"
print D[1]
print "prediction on D:"
print predict(D[0])

fig2 = plt.figure(2)
graph2 = fig2.add_subplot(111)
graph2.plot(w)
graph2.plot(T.zeros_like(w).append(b))
fig2.show()

fig3 = plt.figure(3)
graph3 = fig3.add_subplot(111)
graph3.plot(D[1])
graph3.plot(predict(D[0]))
fig3.show()