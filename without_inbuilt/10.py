import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

class DimReduction:
    def __init__(self, method, n_components):
        self.method = method
        self.n_components = n_components
        self.projection = None

    def fit_transform(self, X, y=None):
        if self.method == 'PCA':
            mean = np.mean(X, axis=0)
            X_centered = X - mean
            cov = np.cov(X_centered.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            self.projection = eigenvectors[:, :self.n_components]
        elif self.method == 'LDA':
            class_labels = np.unique(y)
            mean_overall = np.mean(X, axis=0)
            SW = np.zeros((X.shape[1], X.shape[1]))
            SB = np.zeros((X.shape[1], X.shape[1]))
            for c in class_labels:
                X_c = X[y == c]
                mean_c = np.mean(X_c, axis=0)
                SW += (X_c - mean_c).T.dot((X_c - mean_c))
                n_c = X_c.shape[0]
                mean_diff = (mean_c - mean_overall).reshape(X.shape[1], 1)
                SB += n_c * (mean_diff).dot(mean_diff.T)
            A = np.linalg.inv(SW).dot(SB)
            eigenvalues, eigenvectors = np.linalg.eig(A)
            self.projection = eigenvectors[:, :self.n_components]
        return np.dot(X, self.projection)

def plot_data(X_projected, y, xlabel, ylabel):
    plt.scatter(X_projected[:, 0], X_projected[:, 1], c=y, cmap="jet")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

X, y = load_iris(return_X_y=True)
methods = ['PCA', 'LDA']
for method in methods:
    dr = DimReduction(method, 2)
    X_projected = dr.fit_transform(X, y)
    print(f"Shape of transformed data ({method}):", X_projected.shape)
    plot_data(X_projected, y, f"{method} Component 1", f"{method} Component 2")
