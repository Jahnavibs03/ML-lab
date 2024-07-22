import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# Data for AND function
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])  # A AND B

# Data for OR function
X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([0, 1, 1, 1])  # A OR B

# AND Perceptron
perceptron_and = Perceptron(max_iter=1000,random_state=0)
perceptron_and.fit(X_and, y_and)
y_pred_and = perceptron_and.predict(X_and)
accuracy_and = accuracy_score(y_and, y_pred_and)
print(f"AND Function Accuracy: {accuracy_and * 100}%")
print(f"AND Predictions: {y_pred_and}")

# OR Perceptron
perceptron_or = Perceptron(max_iter=1000, random_state=0)
perceptron_or.fit(X_or, y_or)
y_pred_or = perceptron_or.predict(X_or)
accuracy_or = accuracy_score(y_or, y_pred_or)
print(f"OR Function Accuracy: {accuracy_or * 100}%")
print(f"OR Predictions: {y_pred_or}")
    
