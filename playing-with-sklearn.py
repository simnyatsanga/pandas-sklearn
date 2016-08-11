import numpy as np
import pandas as pd
from pandas import DataFrame
import datetime
import pandas.io.data
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression

class Regression():
    X_train = pd.DataFrame()
    X_test = pd.DataFrame()
    Y_train = pd.DataFrame()
    Y_test = pd.DataFrame()
    wine_df = pd.DataFrame()
    lm = LinearRegression()

    def import_data(self):
        self.wine_df = pd.read_csv('winequality-white.csv')

    def split_data(self):
        self.X_train, self.X_test, self.Y_train, self.Y_test = sklearn.cross_validation.train_test_split(self.wine_df, self.wine_df.quality, test_size=0.33, random_state=5)

    def drop_target_column(self):
        self.X_train.drop('quality', axis=1, inplace=True)
        self.X_test.drop('quality', axis=1, inplace=True)

    def train_model(self):
        self.lm.fit(self.X_train, self.Y_train)

    def test_model(self):
        print "Fit a model X_train, and calculate MSE with Y_train:", np.mean((self.Y_train - self.lm.predict(self.X_train)) **2)
        print "Fit a model X_train, and calculate MSE with X_test, Y_test:", np.mean((self.Y_test - self.lm.predict(self.X_test)) **2)

    def main(self):
        self.import_data()
        self.split_data()
        self.drop_target_column()
        self.train_model()
        self.test_model()

if __name__ == "__main__":
    regression = Regression()
    regression.main()
