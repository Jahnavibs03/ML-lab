import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load Glass dataset
df = pd.read_csv(“glass.csv”)

# Features and target variable
X = df.drop(columns=['Type'])
y = df['Type']

# Function to train and evaluate KNN classifier
def knn(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
    metrics = ['euclidean','manhattan']
    for x in metrics:
       model = KNeighborsClassifier(n_neighbors=3,metric=x)
       model.fit(X_train, y_train)
       y_pred = model.predict(X_test)
       accuracy = accuracy_score(y_test, y_pred)
       print(f"metric={x}: Accuracy = {accuracy:.4f}")

knn(X,y)

