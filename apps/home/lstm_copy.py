from turtle import forward
from matplotlib.pyplot import get
import numpy as np #The Numpy numerical computing library
import pandas as pd #The Pandas data science library
from IPython.display import display
from scraping import stock, history
from list import sample
from statistics import mean
from nsepy import get_history
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import time
import matplotlib.pyplot as plt

# To remove the scientific notation from numpy arrays
np.set_printoptions(suppress=True)

startDate='2020-1-1'
endDate=datetime(2020, 10, 5)

# Getting the Data
get_data=history(stock('AAPL'), startDate)
print(get_data.head())
print(get_data.shape)

get_data['TradeDate']=get_data.index

# Placing the data in variables
FullData=get_data[['Close']].values
LowPrice=get_data[['Low']].values
HighPrice=get_data[['High']].values
ClosePrice=get_data[['Close']].values
OpenPrice=get_data[['Open']].values
Volume=get_data[['Volume']].values

sc=MinMaxScaler()

DataScaler=sc.fit(FullData)
X=DataScaler.transform(FullData)

print('### After Normalization ###')

print('Original Prices')
print(FullData[-10:])

print('##################')

X=X.reshape(X.shape[0],)
print('Scaled Prices')
print(X[-10:])

X_samples = list()
y_samples= list()

NumberofRows=len(X)
TimeSteps=10
FutureTimeSteps=5

for i in range(TimeSteps, NumberofRows-FutureTimeSteps, 1):
    x_sample=X[i-TimeSteps:i]
    y_sample=X[i:i+FutureTimeSteps]
    X_samples.append(x_sample)
    y_samples.append(y_sample)

X_data=np.array(X_samples)
X_data=X_data.reshape(X_data.shape[0],X_data.shape[1],1)
print('### Input Data Shape ###') 
print(X_data.shape)

y_data=np.array(y_samples)
print('### Output Data Shape ###') 
print(y_data.shape)

TestingRecords=5

X_train=X_data[:-TestingRecords]
X_test=X_data[-TestingRecords:]
y_train=y_data[:-TestingRecords]
y_test=y_data[-TestingRecords:]

print('\n#### Training Data shape ####')
print(X_train.shape)
print(y_train.shape)

print('\n#### Testing Data shape ####')
print(X_test.shape)
print(y_test.shape)

for inp, out in zip(X_train[0:2], y_train[0:2]):
    print(inp)
    print('====>')
    print(out)
    print('#'*20)

TimeSteps=X_train.shape[1]
TotalFeatures=X_train.shape[2]
print("Number of TimeSteps:", TimeSteps)
print("Number of Features:", TotalFeatures)

regressor=Sequential()

regressor.add(LSTM(units = 10, activation = 'relu', input_shape = (TimeSteps, TotalFeatures), return_sequences=True))

regressor.add(LSTM(units = 5, activation = 'relu', input_shape = (TimeSteps, TotalFeatures), return_sequences=True))

regressor.add(LSTM(units = 5, activation = 'relu', return_sequences=False ))

regressor.add(Dense(units = FutureTimeSteps))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

StartTime=time.time()

regressor.fit(X_train, y_train, batch_size = 5, epochs = 100)

EndTime=time.time()
print("############### Total Time Taken: ", round((EndTime-StartTime)/60), 'Minutes #############')

predicted_Price = regressor.predict(X_test)
predicted_Price = DataScaler.inverse_transform(predicted_Price)
print('#### Predicted Prices ####')
print(predicted_Price)

orig=y_test
orig=DataScaler.inverse_transform(y_test)
print('\n#### Original Prices ####')
print(orig)