import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# K-means function
def kmeans(X, K, max_iters=100):
    centroids = X[:K]
    for _ in range(max_iters):
        expanded_x = X[:, np.newaxis]
        euc_dist = np.linalg.norm(expanded_x - centroids, axis=2)
        labels = np.argmin(euc_dist, axis=1)
        new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(K)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return labels, centroids

# Load the Iris dataset
X = load_iris().data
K = 3

# Perform K-means clustering
labels, centroids = kmeans(X, K)
print("Labels:", labels)
print("Centroids:", centroids)

# Plot the results
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='red', s=200)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('K-means Clustering of Iris Dataset')
plt.show()
