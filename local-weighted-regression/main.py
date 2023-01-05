# notes
# predict
# theta^T . X
# no training
# parameter theta is calculated (X^T WX)^-1 * (X^TWY)
# theta minimises error
# Also need to calculate weight, which is a diagonal matrix
# Wi = e ^ -1((xi - x)^T . (xi - x) / 2T^2)
# T is possibly manual value
import math
import numpy
import matplotlib.pyplot as plot


def calculate_weight(xi, x, t):
    wi = numpy.exp((numpy.dot((xi - x).T, xi - x)/(-1 * 2 * t * t)))
    return wi


def calculate_theta(X, Y, W):
    wx = X.T * W
    return numpy.linalg.inv(wx @ X) @ wx @ Y


def predict(X, Y, xi, t):
    xi = numpy.array([xi, 1])
    W = numpy.ones(len(X))
    with_bias = numpy.c_[X, numpy.ones(len(X))]

    for i in range(len(X)):
        W[i] = calculate_weight(with_bias[i], xi, t)

    theta = calculate_theta(with_bias, Y, W)

    return theta.T @ xi


X = numpy.linspace(-5, 5, 1000)
Y = X**2
plot.plot(X, Y)
test_x = numpy.linspace(-5, 5, 100)

predictions = []
for xi in test_x:
    predictions.append(predict(X, Y, xi, 12))

test_x = numpy.array(test_x).reshape(100, 1)
predictions = numpy.array(predictions).reshape(100, 1)

plot.plot(test_x, predictions, 'r.')
X = X.reshape(1000, 1)

plot.show()
