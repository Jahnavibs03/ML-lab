import numpy as np
import pandas as pd
from collections import Counter

# Define distance functions
def euc(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def man(x1, x2):
    return np.sum(np.abs(x1 - x2))

# Load the dataset
df = pd.read_csv("glass.csv")
X = df.drop("Type", axis=1).values
y = df['Type'].values

# Shuffle and split the data into training and testing sets (70-30 split)
shf = np.random.permutation(len(X))
split = int(0.7 * len(X))
X_train, X_test = X[shf[:split]], X[shf[split:]]
y_train, y_test = y[shf[:split]], y[shf[split:]]

# KNN function
def knn_predict(X_train, y_train, X_test, distance_fn):
    pred = []
    for x in X_test:
        distances = [distance_fn(x, x_train) for x_train in X_train]
        k_indices = np.argsort(distances)[:3]
        k_labels = [y_train[i] for i in k_indices]
        most_common = Counter(k_labels).most_common(1)[0][0]
        pred.append(most_common)
    return pred

# Predict using Euclidean distance
p1 = knn_predict(X_train, y_train, X_test, distance_fn=euc)
# Predict using Manhattan distance
p2 = knn_predict(X_train, y_train, X_test, distance_fn=man)

# Accuracy function
def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

# Calculate and print accuracies
acc = accuracy(y_test, p1)
print(f"Accuracy using Euclidean distance: {acc:.2f}")
acc = accuracy(y_test, p2)
print(f"Accuracy using Manhattan distance: {acc:.2f}")
