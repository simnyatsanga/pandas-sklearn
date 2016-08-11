import numpy as np
import pandas as pd
from pandas import DataFrame
import datetime
import pandas.io.data
import matplotlib.pyplot as plt
import sklearn

from sklearn.linear_model import LinearRegression
lm = LinearRegression()

wine_df = pd.read_csv('winequality-white.csv')
print wine_df.shape
print wine_df.head()

X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(wine_df, wine_df.quality, test_size=0.33, random_state=5)

print X_train.shape
print X_test.shape
print Y_train.shape
print Y_test.shape
#
X_train.drop('quality', axis=1, inplace=True)
X_test.drop('quality', axis=1, inplace=True)

lm.fit(X_train, Y_train)

print "Fit a model X_train, and calculate MSE with Y_train:", np.mean((Y_train - lm.predict(X_train)) **2)
print "Fit a model X_train, and calculate MSE with X_test, Y_test:", np.mean((Y_test - lm.predict(X_test)) **2)
