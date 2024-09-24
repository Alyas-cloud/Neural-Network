import numpy as np

# Activation function (Sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function for backpropagation
def sigmoid_derivative(x):
    return x * (1 - x)

# Neural Network Class
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden = np.random.randn(1, hidden_size)
        self.bias_output = np.random.randn(1, output_size)

    def feedforward(self, X):
        # Feedforward through the hidden layer
        self.hidden_layer_activation = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = sigmoid(self.hidden_layer_activation)

        # Feedforward through the output layer
        self.output_layer_activation = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.predicted_output = sigmoid(self.output_layer_activation)
        return self.predicted_output

    def backpropagate(self, X, y, learning_rate):
        # Calculate error (difference between predicted and actual output)
        error_output_layer = y - self.predicted_output

        # Calculate gradients for the output layer
        d_predicted_output = error_output_layer * sigmoid_derivative(self.predicted_output)

        # Calculate error for the hidden layer
        error_hidden_layer = d_predicted_output.dot(self.weights_hidden_output.T)

        # Calculate gradients for the hidden layer
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(self.hidden_layer_output)

        # Update weights and biases (backpropagation)
        self.weights_hidden_output += self.hidden_layer_output.T.dot(d_predicted_output) * learning_rate
        self.bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate
        self.bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            # Feedforward
            self.feedforward(X)

            # Backpropagate and update weights and biases
            self.backpropagate(X, y, learning_rate)

            # Print loss for every 100th epoch
            if epoch % 100 == 0:
                loss = np.mean(np.square(y - self.predicted_output))
                print(f'Epoch {epoch} Loss: {loss}')

# Example usage
if __name__ == "__main__":
    # Input data (X) and corresponding output (y)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # XOR input
    y = np.array([[0], [1], [1], [0]])  # XOR output

    # Initialize the neural network
    nn = SimpleNeuralNetwork(input_size=2, hidden_size=2, output_size=1)

    # Train the neural network
    nn.train(X, y, epochs=10000, learning_rate=0.1)

    # Test the neural network after training
    print("\nPredicted Output after training:")
    print(nn.feedforward(X))
