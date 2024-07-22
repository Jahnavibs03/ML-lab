import pandas as pd
import numpy as np
import seaborn as sns

# Load the dataset
df = sns.load_dataset('titanic')

# Preprocess the data
df = df.drop(['who', 'deck', 'embark_town', 'alive', 'class', 'adult_male', 'alone'], axis=1)
df['age'].fillna(df['age'].median(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
df['sex'] = df['sex'].map({'female': 0, 'male': 1}).astype(int)
df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
df = df.dropna()

# Split the data into training and testing sets
split = int(0.8 * len(df))
train = df.iloc[:split]
test = df.iloc[split:]

# Calculate priors
priors = train['survived'].value_counts(normalize=True).to_dict()

# Calculate mean and standard deviation for conditionals
conds = {}
for feat in train.columns[train.columns != 'survived']:
    conds[feat] = {}
    for lbl in train['survived'].unique():
        subset = train[train['survived'] == lbl][feat]
        conds[feat][lbl] = (subset.mean(), subset.std())

# Function to predict using Naive Bayes
def predict(inst):
    post = {lbl: np.log(priors[lbl]) for lbl in priors}
    for feat, stats in conds.items():
        x = inst[feat]
        for lbl, (mean, std) in stats.items():
            if std == 0:
                std = 1e-6
            exp = np.exp(-(x - mean) ** 2 / (2 * std ** 2))
            like = (1 / (np.sqrt(2 * np.pi) * std)) * exp
            post[lbl] += np.log(like)
    return max(post, key=post.get)

# Test the model
y_true = []
y_pred = []
for _, inst in test.iterrows():
    y_true.append(inst['survived'])
    y_pred.append(predict(inst))

# Calculate accuracy
acc = np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)
print("Accuracy:", acc)
