import numpy as np

# Activation function - sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Define XOR inputs and labels
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([[0], [1], [1], [0]])

# Define AND-NOT inputs and labels
X_and_not = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and_not = np.array([[1], [1], [0], [0]])

# Initialize weights and biases
np.random.seed(1)
input_size = X_xor.shape[1]
hidden_size = 4
output_size = 1
w_h = np.random.uniform(size=(input_size, hidden_size))
b_h = np.random.uniform(size=(1, hidden_size))
w_o = np.random.uniform(size=(hidden_size, output_size))
b_o = np.random.uniform(size=(1, output_size))

# Training function for MLP
def train_mlp(X, y, epochs=1000, lr=0.1):
    global w_h, b_h, w_o, b_o
    for epoch in range(epochs):
        # Forward propagation
        h_in = np.dot(X, w_h) + b_h
        h_out = sigmoid(h_in)
        o_in = np.dot(h_out, w_o) + b_o
        o_out = sigmoid(o_in)

        # Backpropagation
        error = y - o_out
        d_o = error * sigmoid_derivative(o_out)
        error_h = d_o.dot(w_o.T)
        d_h = error_h * sigmoid_derivative(h_out)

        # Update weights and biases
        w_o += h_out.T.dot(d_o) * lr
        b_o += np.sum(d_o, axis=0, keepdims=True) * lr
        w_h += X.T.dot(d_h) * lr
        b_h += np.sum(d_h, axis=0, keepdims=True) * lr
    return o_out

# Training XOR MLP
print("Training XOR MLP:")
out_xor = train_mlp(X_xor, y_xor)

# Training AND-NOT MLP
print("\nTraining AND-NOT MLP:")
out_and_not = train_mlp(X_and_not, y_and_not)

# Print final predictions
print("\nFinal predictions for XOR MLP:")
for i in range(len(X_xor)):
    print(f"Input: {X_xor[i]}, Predicted output: {out_xor[i][0]:.4f}")

print("\nFinal predictions for AND-NOT MLP:")
for i in range(len(X_and_not)):
    print(f"Input: {X_and_not[i]}, Predicted output: {out_and_not[i][0]:.4f}")
