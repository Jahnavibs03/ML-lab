import numpy as np

def train_perceptron(X, y, lr=0.1, epochs=100):
    X = np.c_[X, np.ones(len(X))]
    w = np.zeros(X.shape[1])
    for _ in range(epochs):
        for i in range(len(X)):
            pred = np.dot(X[i], w)
            err = y[i] - (1 if pred >= 0 else 0)
            w += lr * err * X[i]
    return w

def predict_perceptron(x, w):
    x = np.append(x, 1)
    pred = np.dot(x, w)
    return 1 if pred >= 0 else 0

# Example usage for AND function
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])
print("Training AND Perceptron:")
w_and = train_perceptron(X_and, y_and)
print("Testing AND Perceptron:")
for x in X_and:
    pred = predict_perceptron(x, w_and)
    print(f"Inputs: {x}, Prediction: {pred}")

# Example usage for OR function
X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([0, 1, 1, 1])
print("\nTraining OR Perceptron:")
w_or = train_perceptron(X_or, y_or)
print("Testing OR Perceptron:")
for x in X_or:
    pred = predict_perceptron(x, w_or)
    print(f"Inputs: {x}, Prediction: {pred}")
