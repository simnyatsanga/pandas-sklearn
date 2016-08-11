import numpy as np
import pandas as pd
from pandas import DataFrame
import datetime
import pandas.io.data
import matplotlib.pyplot as plt
import sklearn

cars_df = pd.read_csv('ToyotaCorolla.csv')
print cars_df.shape
print cars_df.head()

cars_df['FuelType1'] = np.where(cars_df['FuelType'] == 'CNG', 1 , 0)
cars_df['FuelType2'] = np.where(cars_df['FuelType'] == 'Diesel', 1 , 0)
cars_df.drop('FuelType', axis=1, inplace=True)

#Scatter plotting two variables
plt.scatter(cars_df.KM, cars_df.Price)
plt.xlabel("Age")
plt.ylabel("Price")
plt.show()

#Pulling data from Yahoo finance
#sp500 = pd.io.data.get_data_yahoo('%5EGSPC', start = datetime.datetime(2000, 10, 1), end = datetime.datetime(2016, 8, 5))

# Reading csv into dataframe and setting date to be the intuitive x-axis when plotting more than variable
# df = pd.read_csv('sp500.csv', index_col = 'Date',parse_dates = True)

# Difference between columns
# df['H-L'] = df['High'] - df.Low


#  Adding rolling mean column to the df using Close column
# df['100MA'] = pd.rolling_mean(df['Close'], 100, min_periods=1)
#
# Calculating standard deviation
# df['STD'] = pd.rolling_std(df['Close'], 25, min_periods=1)

# Correlation between variables. Useful to get insights on variables that maybe dependent on each other
# print df[['Volume', 'H-L']].corr()
#


# 2D Plotting several columns
# correllationComp[['AAPL', 'MSFT', 'TSLA', 'EBAY', 'SBUX', 'AAPL_Multi']].plot()
# plt.show()
