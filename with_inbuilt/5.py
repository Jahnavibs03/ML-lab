import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load Titanic dataset
df = sns.load_dataset('titanic')

# Preprocessing
df.drop(columns=['deck', 'embark_town', 'alive', 'who', 'class', 'adult_male', 'alone'], inplace=True)
df.dropna(inplace=True)
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['embarked'] = df['embarked'].map({'C': 0, 'Q': 1, 'S': 2})
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

# Features and target variable
X = df.drop(columns=['survived'])
y = df['survived']

# Function to train and evaluate Naive Bayes classifier
def naive_bayes(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy = {accuracy:.4f}")

naive_bayes(X,y)

# Leave-One-Out Cross-Validation
loo = LeaveOneOut()
model = GaussianNB()
accuracies = cross_val_score(model, X, y, cv=loo)
print(f"\nLeave-One-Out Cross-Validation: Accuracy = {np.mean(accuracies):.4f}")

