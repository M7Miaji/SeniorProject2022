from scraping import stock, history
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns   
import math
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, precision_score
from matplotlib.pyplot import axis, get
from scraping import stock, history
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import tensorflow as tf
import keras
from keras import optimizers
from keras.callbacks import History
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate, CuDNNLSTM, BatchNormalization
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint, ModelCheckpoint
from collections import deque
import random
import time
import ta 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import classification_report
from sklearn import metrics
np.set_printoptions(suppress=True)

def applytechnicals(df, feature_names):
    for i in range(2, 10, 2):
        df['RSI_'+str(i)] = ta.momentum.rsi(df['Close'], window=i)
        df['SMA_'+str(i * 10)] = df['Close'].rolling(i*10).mean()
        df['macd']=ta.trend.macd_diff(df['Close'])
        df['EWMA_'+str(i * 10)] = df['Close'].ewm(span=i*10).mean()
        feature_names = feature_names + ['RSI_' + str(i), 'SMA_' + str(i * 10), 'EWMA_' + str(i * 10)]

    df['CMA_30'] = df['Close'].expanding().mean()
    CMA_features = ['CMA_30']

    feature_names.extend(CMA_features)
    df.dropna(inplace=True)

    return feature_names
    

def forest_main():
    startDate='2020-1-1'
    df=history(stock('AAPL'), startDate)
    df.drop('Dividends', axis=1, inplace=True) 
    df.drop('Stock Splits', axis=1, inplace=True)
    df.drop('Volume', axis=1, inplace=True) 
    print(df)
    feature_names = []
    feature_names = applytechnicals(df, feature_names)
    print(df.head())
    random_regressor(df)

def random_regressor(df):

    df = df.select_dtypes(exclude=['object'])
    df=df.fillna(df.mean())
    X = df.drop('Close',axis=1)
    y = df['Close']

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    train_size = int(0.7 * y.shape[0])
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]

    regressor = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)

    ac=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
    print(ac.head())

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    # Calculate the absolute errors
    errors = abs(y_pred - y_test)
    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / y_test)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')

def lstm_model(TimeSteps, TotalFeatures, FutureTimeSteps):
    
    model=Sequential()
    model.add(LSTM(units = 10, activation = 'relu', input_shape = (TimeSteps, TotalFeatures), return_sequences=True))
    model.add(LSTM(units = 5, activation = 'relu', input_shape = (TimeSteps, TotalFeatures), return_sequences=True))
    model.add(LSTM(units = 5, activation = 'relu', return_sequences=False ))
    model.add(Dense(units = FutureTimeSteps))
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    
    return model

def fit(X_train, y_train, model):

    StartTime=time.time()
    model.fit(X_train, y_train, batch_size = 5, epochs = 100)
    EndTime=time.time()
    print("############### Total Time Taken: ", round((EndTime-StartTime)/60), 'Minutes #############')
    
    return model

def prediction(model, X_test, DataScaler, y_test):

    predicted_Price = model.predict(X_test)
    predicted_Price = DataScaler.inverse_transform(predicted_Price)
    orig=y_test
    orig=DataScaler.inverse_transform(y_test)

    return predicted_Price, orig

def processing(data):

    sc=MinMaxScaler()

    DataScaler=sc.fit(data)

    X=DataScaler.transform(data)
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

    TestingRecords=5

# Splitting the data into train and test
    X_train=X_data[:-TestingRecords]
    X_test=X_data[-TestingRecords:]
    y_train=y_data[:-TestingRecords]
    y_test=y_data[-TestingRecords:]

    for inp, out in zip(X_train[0:2], y_train[0:2]):
        print(inp)
        print('====>')
        print(out)
        print('#'*20)
        
    TimeSteps=X_train.shape[1]
    TotalFeatures=X_train.shape[2]

    return TimeSteps, TotalFeatures, FutureTimeSteps, X_train, y_train, X_test, y_test, DataScaler

def predict_future(model, data, DataScaler):

    Last10DaysPrices=np.array([167.22999573, 166.41999817, 161.78999329, 162.88000488, 156.80000305, 166.41999817, 161.78999329, 162.88000488, 156.80000305, 156.57000732])

# Reshaping the data to (-1,1 )because its a single entry
    Last10DaysPrices=Last10DaysPrices.reshape(-1, 1)

    # [161.78999329 162.88000488 156.80000305 156.57000732 163.63999939]

    # Scaling the data on the same level on which model was trained
    X_test=DataScaler.transform(Last10DaysPrices)

    NumberofSamples=1
    TimeSteps=X_test.shape[0]
    NumberofFeatures=X_test.shape[1]
    # Reshaping the data as 3D input
    X_test=X_test.reshape(NumberofSamples,TimeSteps,NumberofFeatures)

    # Generating the predictions for next 5 days
    Next5DaysPrice = model.predict(X_test)

    # Generating the prices in original scale
    Next5DaysPrice = DataScaler.inverse_transform(Next5DaysPrice)
    print(Next5DaysPrice)

def main():
    startDate='2010-1-1'
    get_data=history(stock('AAPL'), startDate)
    get_data['TradeDate']=get_data.index
    get_data.drop('Dividends', axis=1, inplace=True) 
    get_data.drop('Stock Splits', axis=1, inplace=True)
    get_data.drop('Volume', axis=1, inplace=True)  
    tech = applytechnicals(get_data)
    print(tech)
    print(get_data.shape)
    data =get_data[['Close']].values

    TimeSteps, TotalFeatures, FutureTimeSteps, X_train, y_train, X_test, y_test, DataScaler = processing(data) # Check
   
    model = lstm_model(TimeSteps, TotalFeatures, FutureTimeSteps)

    model = fit(X_train, y_train, model)

    predicted_Price, orig = prediction(model, X_test, DataScaler, y_test)

    
    print("Original Price")
    print(orig)
    print("Predicted Price")
    print(predicted_Price)
    
    for i in range(len(orig)):
        Prediction=predicted_Price[i]
        Original=orig[i]
    print(Prediction)
    print(Original)
    print(str(100 - (100*(abs(Original-Prediction)/Original)).mean().round(2)))

    predict_future(model, get_data, DataScaler)
    

forest_main()