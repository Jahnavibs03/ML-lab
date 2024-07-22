import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

def plot_data(X_projected, y, xlabel, ylabel):
    plt.scatter(X_projected[:, 0], X_projected[:, 1], c=y, cmap="jet")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# Load the Iris dataset
X, y = load_iris(return_X_y=True)

# Perform data preprocessing - Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define the number of components
n_components = 2

# PCA
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)
print(f"Shape of transformed data (PCA): {X_pca.shape}")
print(f"Transformed data: {X_pca}")
plot_data(X_pca, y, "PCA Component 1", "PCA Component 2")

# LDA
lda = LinearDiscriminantAnalysis(n_components=n_components)
X_lda = lda.fit_transform(X_scaled, y)
print(f"Shape of transformed data (LDA): {X_lda.shape}")
print(f"Transformed data: {X_lda}")
plot_data(X_lda, y, "LDA Component 1", "LDA Component 2")

