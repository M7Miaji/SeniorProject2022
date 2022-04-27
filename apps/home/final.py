from scraping import stock, history
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns   
import math
from sklearn.metrics import mean_squared_error
from matplotlib.pyplot import get
from scraping import stock, history
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import tensorflow as tf
import keras
from keras import optimizers
from keras.callbacks import History
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
import time

np.set_printoptions(suppress=True)
startDate='2020-1-1'
FullData=history(stock('AAPL'), startDate)
FullData['TradeDate']=FullData.index

def lstm_model(TimeSteps, TotalFeatures, FutureTimeSteps):
    regressor=Sequential()
    regressor.add(LSTM(units = 10, activation = 'relu', input_shape = (TimeSteps, TotalFeatures), return_sequences=True))
    regressor.add(LSTM(units = 5, activation = 'relu', input_shape = (TimeSteps, TotalFeatures), return_sequences=True))
    regressor.add(LSTM(units = 5, activation = 'relu', return_sequences=False ))
    regressor.add(Dense(units = FutureTimeSteps))
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    return regressor

def fit(X_train, y_train, regressor):
    StartTime=time.time()
    regressor.fit(X_train, y_train, batch_size = 5, epochs = 100)
    EndTime=time.time()
    print("############### Total Time Taken: ", round((EndTime-StartTime)/60), 'Minutes #############')
    return regressor

def prediction(regressor, X_test, DataScaler, y_test):
    predicted_Price = regressor.predict(X_test)
    predicted_Price = DataScaler.inverse_transform(predicted_Price)
    orig=y_test
    orig=DataScaler.inverse_transform(y_test)
    return predicted_Price, orig

def processing(Data):
    sc=MinMaxScaler()
    DataScaler=sc.fit(Data)
    X=DataScaler.transform(Data)
    X=X.reshape(X.shape[0],)
    X_samples=list()
    y_samples=list()
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
    y_data=np.array(y_samples)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.30, random_state = 0)
    for inp, out in zip(X_train[0:2], y_train[0:2]):
        print(inp)
        print('====>')
        print(out)
        print('#'*20)
    TimeSteps=X_train.shape[1]
    TotalFeatures=X_train.shape[2]
    return TimeSteps, TotalFeatures, FutureTimeSteps, X_train, y_train, X_test, y_test, DataScaler

def main():
    startDate='2020-1-1'
    get_data=history(stock('AAPL'), startDate)
    get_data['TradeDate']=get_data.index
    data=get_data[['Close']].values
    random_forest_data = pd.DataFrame()
    print(get_data.head())
    TimeSteps, TotalFeatures, FutureTimeSteps, X_train, y_train, X_test, y_test, DataScaler = processing(data)
    regressor = lstm_model(TimeSteps, TotalFeatures, FutureTimeSteps)
    regressor = fit(X_train, y_train, regressor)
    predicted_Price, orig = prediction(regressor, X_test, DataScaler, y_test)
    print("original price")
    print(orig)
    print("predicted price")
    print(predicted_Price)
    for i in range(len(orig)):
        Prediction=predicted_Price[i]
        Original=orig[i]
    print(str(100 - (100*(abs(Original-Prediction)/Original)).mean().round(2)))


main()