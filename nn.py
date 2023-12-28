"""

    A python implementation of a neural network with one hidden layer using Matrix operations.


    source: https://datascience.stackexchange.com/questions/75855/what-types-of-matrix-multiplication-are-used-in-machine-learning-when-are-they

"""
import numpy as np

error_rate = 0.1

# Activation function
def sigmoid(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

# Feed forward one layer
def feed_forward(inputs, weights, biases):
    layer_input = np.dot(inputs, weights) + biases
    return sigmoid(layer_input)

# Back propagation
def back_propagation(inputs, outputs, weights, biases, learning_rate):
    # Feed forward
    hidden_layer_output = feed_forward(inputs, weights[0], biases[0])
    final_output = feed_forward(hidden_layer_output, weights[1], biases[1])

    # Calculate error
    error = outputs - final_output

    # Calculate deltas
    d_output = error * sigmoid(final_output, True)
    error_hidden_layer = d_output.dot(weights[1].T)
    d_hidden_layer = error_hidden_layer * sigmoid(hidden_layer_output, True)

    # Update weights and biases
    weights[1] += hidden_layer_output.T.dot(d_output) * learning_rate
    weights[0] += inputs.T.dot(d_hidden_layer) * learning_rate
    biases[1] += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    biases[0] += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    return error

# Training function
def train(inputs, outputs, learning_rate):
    input_layer_neurons, hidden_layer_neurons, output_neurons = 2, 2, 1

    # Initialize weights and biases
    weights = [
        np.random.rand(input_layer_neurons, hidden_layer_neurons),
        np.random.rand(hidden_layer_neurons, output_neurons)
    ]
    biases = [
        np.random.rand(1, hidden_layer_neurons),
        np.random.rand(1, output_neurons)
    ]

    # Train while error is is greater than error_rate
    epoch = 0
    while True:
        error = back_propagation(inputs, outputs, weights, biases, learning_rate)
        if epoch % 1000 == 0:
            print(f'Epoch: {epoch}, Error: {np.mean(np.abs(error))}')
        if np.mean(np.abs(error)) < error_rate:
            break
        epoch += 1

    return weights, biases

# Prediction function
def predict(inputs, weights, biases):
    hidden_layer_output = feed_forward(inputs, weights[0], biases[0])
    final_output = feed_forward(hidden_layer_output, weights[1], biases[1])
    return final_output

# Main function
def main():
    # XOR problem
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    outputs = np.array([[0], [1], [1], [0]])

    learning_rate = 0.1

    weights, biases = train(inputs, outputs, learning_rate)

    # Test
    for input in inputs:
        print(f'Input: {input}, Predicted Output: {predict(np.array([input]), weights, biases)}')

if __name__ == "__main__":
    main()
