from random import random
import pandas as pd
import numpy as np

from sklearn.datasets import make_classification
from sklearn import tree
from sklearn.model_selection import train_test_split

# generate regression random data with 100 features
X, y = make_classification(n_informative=4, n_classes=4)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)


model = tree.DecisionTreeClassifier().fit(X_train, y_train)
predictions = model.predict(X_test)

x_train_data = pd.DataFrame(X_train)
x_test_data = pd.DataFrame(X_test)
y_train_data = pd.DataFrame(y_train)
y_test_data = pd.DataFrame(y_test)
predictions_results = pd.DataFrame(predictions)


# output files
x_train_data.to_csv('data/x_train.csv', index=False)
x_test_data.to_csv('data/x_test.csv', index=False)
y_train_data.to_csv('data/y_train.csv', index=False)
y_test_data.to_csv('data/y_test.csv', index=False)
predictions_results.to_csv('data/predictions.csv', index=False)