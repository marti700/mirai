from random import random
import pandas as pd
import numpy as np

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# random_noise
noise = random()
#random_bias (intercept)
bias = random()
# generate regression random data with 100 features
X, y = make_regression(n_features=10, noise=noise, bias=bias)

# Cross validation score with 10 folds cross validation
cross_val_results = cross_val_score(LinearRegression(),X,y, scoring='neg_mean_squared_error', cv=10) 

# Convert to data frame
cross_val_results_DF = pd.DataFrame(cross_val_results)
X_DF = pd.DataFrame(X)
y_DF = pd.DataFrame(y)

# Export to csv file
cross_val_results_DF.to_csv('data/cross_val_scores.csv', index=False)
X_DF.to_csv('data/X_train.csv', index=False)
y_DF.to_csv('data/y_train.csv', index=False)