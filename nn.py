import numpy as np


class Node:
    def __init__(self):
        self.inputs = []
        self.output = None

    def forward(self):
        # Must implement forward pass
        raise NotImplementedError
    
    def backward(self):
        # Implement backpropagation in the subclasses
        pass


class Add(Node):
    def forward(self, x, y):
        self.inputs = [x, y]
        self.output = x + y
        return self.output
    
    def backward(self, output_gradient):
        input_gradient = output_gradient
        biases_gradient = np.sum(output_gradient, axis=0)
        return input_gradient, biases_gradient


class MatMul(Node):
    def forward(self, x, w):
        self.inputs = [x, w]
        self.output = np.dot(x, w)
        return self.output
    
    def backward(self, output_gradient):
        # Reshape the output_gradient to 2D if it's a 1D array
        if output_gradient.ndim == 1:
            output_gradient = output_gradient.reshape(1, -1)

        input_gradient = np.dot(output_gradient, self.inputs[1].T)
        weights_gradient = np.dot(self.inputs[0].reshape(1, -1).T, output_gradient)

        # Ensure the gradients have the correct shape        
        if input_gradient.shape[0] == 1:
            input_gradient = input_gradient.flatten()

        return input_gradient, weights_gradient


class ReLU(Node):
    def forward(self, x):
        self.inputs = [x]
        self.output = np.maximum(0, x)
        return self.output
    
    def backward(self, output_gradient):
        input_gradient = output_gradient * (self.inputs[0] > 0)
        return input_gradient


class SimpleDenseLayer:
    def __init__(self, input_size, output_size):
        # Initialize weights and biases
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.random.randn(output_size)
        self.matmul = MatMul()
        self.add = Add()
        self.relu = ReLU()

    def forward(self, x):
        x = self.matmul.forward(x, self.weights)
        x = self.add.forward(x, self.biases)
        x = self.relu.forward(x)
        return x
    
    def backward(self, output_gradient, learning_rate):
        relu_gradient = self.relu.backward(output_gradient)
        add_gradient, biases_gradient = self.add.backward(relu_gradient)
        print('add_gradient:', add_gradient)
        input_gradient, weights_gradient = self.matmul.backward(add_gradient)

        # Update weights and biases
        # print('here', self.weights, learning_rate, weights_gradient)
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * biases_gradient

        return input_gradient
    

class NeuralNetwork:
    def __init__(self, layer_sizes):
        """
        Initialize the neural network with the given layer sizes.

        :param layer_sizes: A list of integers where each represents the
        number of neurons in that layer.
        """
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(SimpleDenseLayer(layer_sizes[i], layer_sizes[i+1]))

    def forward_pass(self, x):
        """
        Perform a forward pass through the network.

        :param x: Input tensor
        :return: Output tensor after passing through the network
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward_pass(self, output_gradient, learning_rate=0.01):
        i = 0
        for layer in reversed(self.layers):
            print('here: ', i)
            output_gradient = layer.backward(output_gradient, learning_rate)
            i += 1

        return output_gradient


# NOTE: The first element of layer_sizes must match randn's argument here for input_data
layer_sizes = [3, 5, 4, 1]
input_data = np.random.randn(layer_sizes[0])
nn = NeuralNetwork(layer_sizes)

# Example input
output = nn.forward_pass(input_data)
print('forward pass: ', output)

output_gradient = np.random.randn(1)
output = nn.backward_pass(output_gradient)
print('backpropagation gradient: ', output)
