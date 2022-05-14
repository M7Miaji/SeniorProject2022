from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns   
import math
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, precision_score
from matplotlib.pyplot import axis, get
#from scraping import stock, history
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import keras
from keras import optimizers
from keras.callbacks import History
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate, CuDNNLSTM, BatchNormalization
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
import yfinance as yf 

def stock(stock_symbol):
    stock_info = yf.Ticker(stock_symbol)
    return stock_info

def history(stock_ticker, startDate):
	hist = stock_ticker.history(start=startDate)
	return hist
    
np.set_printoptions(suppress=True)

def CCI(df, ndays): 
    df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3 
    df['sma'] = df['TP'].rolling(ndays).mean()
    df['mad'] = df['TP'].rolling(ndays).apply(lambda x: pd.Series(x).mad())
    df['CCI'] = (df['TP'] - df['sma']) / (0.015 * df['mad']) 
    return df

def EVM(df, ndays): 
    dm = ((df['High'] + df['Low'])/2) - ((df['High'].shift(1) + df['Low'].shift(1))/2)
    br = (df['Volume'] / 100000000) / ((df['High'] - df['Low']))
    EVM = dm / br 
    EVM_MA = pd.Series(EVM.rolling(ndays).mean(), name = 'EVM_'+str(ndays)) 
    df = df.join(EVM_MA) 
    return df

# Simple Moving Average 
def SMA(df, ndays): 
    SMA = pd.Series(df['Close'].rolling(ndays).mean(), name = 'SMA_'+str(ndays)) 
    df = df.join(SMA) 
    return df

def EWMA(df, ndays): 
    EMA = pd.Series(df['Close'].ewm(span = ndays, min_periods = ndays - 1).mean(), 
                    name = 'EWMA_' + str(ndays)) 
    df = df.join(EMA) 
    return df 

def ROC(df,n):
    N = df['Close'].diff(n)
    D = df['Close'].shift(n)
    ROC = pd.Series(N/D,name='Rate of Change_'+str(n))
    df = df.join(ROC)
    return df 

def BBANDS(df, n):
    MA = df.Close.rolling(window=n).mean()
    SD = df.Close.rolling(window=n).std()
    df['UpperBB_'+str(n)] = MA + (2 * SD) 
    df['LowerBB_'+str(n)] = MA - (2 * SD)
    return df

def ForceIndex(df, ndays): 
    FI = pd.Series(df['Close'].diff(ndays) * df['Volume'], name = 'ForceIndex_'+str(ndays)) 
    df = df.join(FI) 
    return df

def technicals(df):
    df = CCI(df, 14)
    for i in range(2, 10, 2):
        df = EVM(df, i*10)
        df = SMA(df,i*10)
        df = EWMA(df,i*10)
        df = ROC(df,i*10)
        df = BBANDS(df, i*10)
        df = ForceIndex(df,i)
        df['RSI_'+str(i)] = ta.momentum.rsi(df['Close'], window=i)
        df['macd']=ta.trend.macd_diff(df['Close'])
    ema10 = df['Close'].ewm(span=10).mean()
    ema30 = df['Close'].ewm(span=30).mean()
    df['EMA10gtEMA30'] = np.where(ema10 > ema30, 1, -1)
    # Calculate where Close is > EMA10
    df['ClGtEMA10'] = np.where(df['Close'] > ema10, 1, -1)
    high14= df['High'].rolling(14).max()
    low14 = df['Low'].rolling(14).min()
    df['%K'] = (df['Close'] - low14)*100/(high14 - low14)
    # Williams Percentage Range
    df['%R'] = -100*(high14 - df['Close'])/(high14 - low14)
    days = 6
    # Price Rate of Change
    ct_n = df['Close'].shift(days)
    df['PROC'] = (df['Close'] - ct_n)/ct_n
    df['Return'] = df['Close'].pct_change(1).shift(-1)
    df['%Volume'] = df['Volume'].pct_change(1).shift(-1)
    df['class'] = np.where(df['Return'] > 0, 1, 0)
    df['CMA_30'] = df['Close'].expanding().mean()
    df.dropna(inplace=True)
    df.drop('sma', axis=1, inplace=True)
    df.drop('mad', axis=1, inplace=True)
    return df


def lstm_model(TimeSteps, TotalFeatures, FutureTimeSteps):
    
    model=Sequential()
    model.add(LSTM(units = 10, activation = 'relu', input_shape = (TimeSteps, TotalFeatures), return_sequences=True))
    model.add(LSTM(units = 5, activation = 'relu', input_shape = (TimeSteps, TotalFeatures), return_sequences=True))
    model.add(LSTM(units = 5, activation = 'relu', return_sequences=False ))
    model.add(Dense(units = FutureTimeSteps))
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    '''
    model.add(LSTM(units = 64, activation='relu', input_shape=(TimeSteps, TotalFeatures), return_sequences=True))
    model.add(Dropout(0.3)) # with probability of 0.3
        
    model.add(LSTM(units = 64, activation='relu', input_shape=(TimeSteps, TotalFeatures), return_sequences=True))
    model.add(Dropout(0.3)) # with probability of 0.3
    
    model.add(LSTM(units=32, activation='relu', input_shape=(TimeSteps, TotalFeatures), return_sequences=False))
    model.add(Dropout(0.3)) # with probability of 0.3

    model.add(Dense(units=32,kernel_initializer="uniform",activation='relu'))  
          
    model.add(Dense(units=FutureTimeSteps,kernel_initializer="uniform",activation='linear'))
    '''
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

    TestingRecords=200

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
    Last10DaysPrices=np.array(data[-10:])
    #Last10DaysPrices=np.array([167.22999573, 166.41999817, 161.78999329, 162.88000488, 156.80000305, 166.41999817, 161.78999329, 162.88000488, 156.80000305, 156.57000732])

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
    return Next5DaysPrice

def main_lstm(stockName):
    StartTime=time.time()
    startDate='2020-1-1'
    get_data=history(stock(stockName), startDate)
    get_data['TradeDate']=get_data.index
    get_data.drop('Dividends', axis=1, inplace=True) 
    get_data.drop('Stock Splits', axis=1, inplace=True)
    #get_data.drop('Volume', axis=1, inplace=True)  
    data = get_data[['Close']].values

    print(data.shape)
    TimeSteps, TotalFeatures, FutureTimeSteps, X_train, y_train, X_test, y_test, DataScaler = processing(data) # Check
    print("X train: ", X_train.shape, X_test.shape)
    model = lstm_model(TimeSteps, TotalFeatures, FutureTimeSteps)

    model = fit(X_train, y_train, model)

    predicted_Price, orig = prediction(model, X_test, DataScaler, y_test)

    
    print("Original Price")
    print(orig)
    print("Predicted Price")
    print(predicted_Price)

    array_per = np.empty(0)
    array_org = np.empty(0)
    for i in range(len(orig)):
        Prediction=predicted_Price[i]
        Original=orig[i]
        array_per = np.append(array_per, predicted_Price[i][4])
        array_org = np.append(array_org, orig[i][4])
    print(Prediction)
    print(Original)
    
    accuracy = str(100 - (100*(abs(array_org-array_per)/array_org)).mean().round(2))
    print(accuracy)

    EndTime=time.time()
    len_time = round((EndTime-StartTime)/60)
    #print(len(X_train))
    Next5Days=predict_future(model, array_org, DataScaler)
    print("Test 1")
    print(array_org)

    df = pd.DataFrame()
    df = get_data.tail(281)

    #df = technicals(df)
    #df['Close Prediction'] = array_per.tolist()
    buy = [0]
    sell = [0]
    signal = 'Hold' 
    count = 1
    print("kyky --------------------------------------------------------------")
    if df['Close'].tail(1) > Next5Days[0]:
        sell.append(count)
    elif df['Close'].tail(1) < Next5Days[0]:
        buy.append(count)

    if buy[-1] > sell[-1]:
        signal = "Buy"
    elif buy[-1] < sell[-1]:
        signal = "Sell" 
    print(signal)  

    return array_per, array_org, accuracy, len(X_train), len(X_test), len_time, Next5Days, df
    #predict_future(model, get_data, DataScaler)

array_per, array_org, accuracy, X_train, X_test, len_time, Next5Days, df= main_lstm('AAPL')

#print(df.tail())
#print(df.shape)
#print(df.columns)