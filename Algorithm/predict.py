import sys # Import the library that does system activities (like install other packages)

tickerSymbol = "AMZN"

# Choose date range - format should be 'YYYY-MM-DD' 
startDate = '2015-04-01' # as strings
endDate = '2022-01-01' # as strings

import yfinance as yf

tickerData = yf.Ticker(tickerSymbol)

# Create historic data dataframe and fetch the data for the dates given. 
df = tickerData.history(start = startDate, end = endDate)

print('-----------------------')
print('Done!')
print(df.head())

import pandas as pd 
import numpy as np

date_change = '%Y-%m-%d'

# Create a new date column from the index
df['Date'] = df.index

# Perform the date type change
df['Date'] = pd.to_datetime(df['Date'], format = date_change)

# Create a variable that is the date column
Dates = df['Date']

# Add financial information and indicators 
from ta import add_all_ta_features 
df = add_all_ta_features(df, "Open", "High", "Low", "Close", "Volume", fillna=True) 

print(df.columns)

# Install fastai to use the date function

import fastai.tabular
"""After it is imported, let's add the new date features. """

# Define the date parts 
fastai.tabular.add_datepart(df,'Date', drop = 'True')
fastai.tabular.add
# Ensure the correct format
df['Date'] = pd.to_datetime(df.index.values, format = date_change)

# Add the date parts
fastai.tabular.add_cyclic_datepart(df, 'Date', drop = 'True')

shifts = [1,5,10]

train_pct = .75

w = 16 # width
h = 4 # height

def CorrectColumnTypes(df):

  for col in df.columns[1:80]:
      df[col] = df[col].astype('float')

  for col in df.columns[-10:]:
      df[col] = df[col].astype('float')

  # Categories 
  for col in df.columns[80:-10]:
      df[col] = df[col].astype('category')

  return df

def CreateLags(df,lag_size):

  shiftdays = lag_size
  shift = -shiftdays
  df['Close_lag'] = df['Close'].shift(shift)
  return df, shift

def SplitData(df, train_pct, shift):

  train_pt = int(len(df)*train_pct)
  
  train = df.iloc[:train_pt,:]
  test = df.iloc[train_pt:,:]
  
  x_train = train.iloc[:shift,1:-1]
  y_train = train['Close_lag'][:shift]
  x_test = test.iloc[:shift,1:-1]
  y_test = test['Close'][:shift]

  return x_train, y_train, x_test, y_test, train, test


import plotly.graph_objs as go  # Import the graph ojbects

def PlotModelResults_Plotly(train, test, pred, ticker, w, h, shift_days,name):

  D1 = go.Scatter(x=train.index,y=train['Close'],name = 'Train Actual') # Training actuals
  D2 = go.Scatter(x=test.index[:shift],y=test['Close'],name = 'Test Actual') # Testing actuals
  D3 = go.Scatter(x=test.index[:shift],y=pred,name = 'Our Prediction') # Testing predction
 
  line = {'data': [D1,D2,D3],
          'layout': {
              'xaxis' :{'title': 'Date'},
              'yaxis' :{'title': '$'},
              'title' : name + ' - ' + tickerSymbol + ' - ' + str(shift_days)
          }}
  fig = go.Figure(line)

  fig.show()

from sklearn.metrics import mean_squared_error # Install error metrics 
from sklearn.linear_model import LinearRegression # Install linear regression model
from sklearn.neural_network import MLPRegressor # Install ANN model 
from sklearn.preprocessing import StandardScaler # to scale for ann

def LinearRegression_fnc(x_train,y_train, x_test, y_test):

  lr = LinearRegression()
  lr.fit(x_train,y_train)
  lr_pred = lr.predict(x_test)
  lr_MSE = mean_squared_error(y_test, lr_pred)
  lr_R2 = lr.score(x_test, y_test)
  print('Linear Regression R2: {}'.format(lr_R2))
  print('Linear Regression MSE: {}'.format(lr_MSE))

  return lr_pred

# ANN Function 

def ANN_func(x_train,y_train, x_test, y_test):

  # Scaling data
  scaler = StandardScaler()
  scaler.fit(x_train)
  x_train_scaled = scaler.transform(x_train)
  x_test_scaled = scaler.transform(x_test)


  MLP = MLPRegressor(random_state=1, max_iter=1000, hidden_layer_sizes = (100,), activation = 'identity',learning_rate = 'adaptive').fit(x_train_scaled, y_train)
  MLP_pred = MLP.predict(x_test_scaled)
  MLP_MSE = mean_squared_error(y_test, MLP_pred)
  MLP_R2 = MLP.score(x_test_scaled, y_test)

  print('Muli-layer Perceptron R2 Test: {}'.format(MLP_R2))
  print('Multi-layer Perceptron MSE: {}'.format(MLP_MSE))

  return MLP_pred

def CalcProfit(test_df,pred,j):
  pd.set_option('mode.chained_assignment', None)
  test_df['pred'] = np.nan
  test_df['pred'].iloc[:-j] = pred
  test_df['change'] = test_df['Close_lag'] - test_df['Close'] 
  test_df['change_pred'] = test_df['pred'] - test_df['Close'] 
  test_df['MadeMoney'] = np.where(test_df['change_pred']/test_df['change'] > 0, 1, -1) 
  test_df['profit'] = np.abs(test['change']) * test_df['MadeMoney']
  profit_dollars = test['profit'].sum()
  print('Would have made: $ ' + str(round(profit_dollars,1)))
  profit_days = len(test_df[test_df['MadeMoney'] == 1])
  print('Percentage of good trading days: ' + str( round(profit_days/(len(test_df)-j),2))     )

  return test_df, profit_dollars

for j in shifts: 
  print(str(j) + ' days out:')
  print('------------')
  df_lag, shift = CreateLags(df,j)
  df_lag = CorrectColumnTypes(df_lag)
  x_train, y_train, x_test, y_test, train, test = SplitData(df, train_pct, shift)

  # Linear Regression
  print("Linear Regression")
  lr_pred = LinearRegression_fnc(x_train,y_train, x_test, y_test)
  test2, profit_dollars = CalcProfit(test,lr_pred,j)
  PlotModelResults_Plotly(train, test, lr_pred, tickerSymbol, w, h, j, 'Linear Regression')

  # Artificial Neuarl Network 
  print("ANN")
  MLP_pred = ANN_func(x_train,y_train, x_test, y_test)
  test2, profit_dollars = CalcProfit(test,MLP_pred,j)
  PlotModelResults_Plotly(train, test, MLP_pred, tickerSymbol, w, h, j, 'ANN')
  print('------------')