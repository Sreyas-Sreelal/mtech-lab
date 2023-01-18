import numpy

def activation(x):
    return 1 / (1 + numpy.exp(-x))

def derivative(x):
    return x * (1-x)

X = numpy.array(([23,12], [20, 11], [13, 23],[31,14]))
Y = numpy.array(([23], [45], [67],[80]))
X = X/numpy.amax(X,axis=0)
Y = Y/100
iterations = 50000
learning_rate = 0.5
input_neurons = 2
hidden_neurons = 2
output_neurons = 1

weights_hidden = numpy.random.uniform(size=(input_neurons,hidden_neurons))
bias_hidden = numpy.random.uniform(size=(1,hidden_neurons))
weights_output = numpy.random.uniform(size=(hidden_neurons,output_neurons))
bias_output = numpy.random.uniform(size=(1,output_neurons))

for _ in range(iterations):
    hidden_layer_input = numpy.dot(X,weights_hidden)
    hidden_layer_input += bias_hidden
    activated_hidden = activation(hidden_layer_input)
    
    output_layer_input = numpy.dot(activated_hidden,weights_output)
    output_layer_input += bias_output
    activated_output = activation(output_layer_input)

    error_output = Y - activated_output
    output_gradient = derivative(activated_output)
    d_output = error_output*output_gradient
    
    error_hidden = d_output.dot(weights_output.T)
    hidden_gradient = derivative(activated_hidden)
    d_hidden = error_hidden * hidden_gradient

    weights_output += activated_hidden.T.dot(d_output) * learning_rate
    weights_hidden +=  X.T.dot(d_hidden) * learning_rate

    print("Predicted :",activated_output)
    print("Actual:",Y)
    print()