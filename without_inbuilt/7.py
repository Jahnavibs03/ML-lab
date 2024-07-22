from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('weather_forecast.csv')

# Convert categorical variables to numerical
le = preprocessing.LabelEncoder()
for column in df.columns:
    df[column] = le.fit_transform(df[column])

# Features and target variable
X = df.drop(columns=['Play'])
y = df['Play']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Predictions on testing set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Plot decision tree
plt.figure(figsize=(10, 6))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=['No', 'Yes'])
plt.show()
