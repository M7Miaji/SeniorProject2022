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

def x_y(df, period):
    y = df['Close'].shift(-1) - df['Close']
    y = y[:-period]
    x = df[:-period]
    y = np.where(y > 0, 1, 0)
    return x, y

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
    #clf = split_train(df, feature_names)
    '''
    feature_names = []

    feature_names = applytechnicals(df, feature_names)
    
    print(df)

    X = df[feature_names]
    y = df['Close']

    train_size = int(0.85 * y.shape[0])
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]

    rf_model = RandomForestRegressor(n_estimators=200, max_depth=3, max_features=4, random_state=42)
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)

    y_pred_series = pd.Series(y_pred, index=y_test.index)

    print(y_pred)
    '''

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


def split_train(df, feature_names):
    print(feature_names)
    x = df[feature_names].fillna(0)
    y = df['Close'].fillna(0)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.33, random_state=42)

    clf = RandomForestClassifier(n_estimators=500, criterion='gini', max_depth=20, min_samples_leaf=10, n_jobs=1, warm_start=True)

    clf = clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    report = classification_report(y_test, y_pred)

    print(report)

    return clf

def predict_timeseries(df, clf):

    df['Buy'] = df['Close']

    for i in range(len(df)):
        X_cls_valid = [[df['aboveSAR'][i],df['aboveUpperBB'][i],df['belowLowerBB'][i],
                        df['RSI'][i],df['oversoldRSI'][i],df['overboughtRSI'][i],
                        df['aboveEMA5'][i],df['aboveEMA10'][i],df['aboveEMA15'][i],df['aboveEMA20'][i],
                        df['aboveEMA30'][i],df['aboveEMA40'][i],df['aboveEMA50'][i],
                        df['aboveEMA60'][i],df['aboveEMA70'][i],df['aboveEMA80'][i],df['aboveEMA90'][i],
                        df['aboveEMA100'][i]]]    

        y_cls_pred_valid = clf.predict(X_cls_valid)
        df['Buy'][i] = y_cls_pred_valid[0].copy()

    print(df.head())
    return df



    return x_train, x_test, y_train, y_test
def lstm_model(TimeSteps, TotalFeatures, FutureTimeSteps):
    
    model=Sequential()
    model.add(LSTM(units = 10, activation = 'relu', input_shape = (TimeSteps, TotalFeatures), return_sequences=True))
    model.add(LSTM(units = 5, activation = 'relu', input_shape = (TimeSteps, TotalFeatures), return_sequences=True))
    model.add(LSTM(units = 5, activation = 'relu', return_sequences=False ))
    model.add(Dense(units = FutureTimeSteps))
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    
    return model

def reg_model():
    model = 0
    return model

def fit(X_train, y_train, model):

    StartTime=time.time()
    #model.fit(X_train, y_train, batch_size = 5, epochs = 10)
    #model.fit(X_train, y_train, batch_size = 5, epochs = 50)
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

    #data=Data[['Close']].values

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

def process_test(data):
    #sc=MinMaxScaler(feature_range=(0, 1))
    #DataScaler=sc.fit(data)
    x = data[['Mean','Close']].values
    scaler = MinMaxScaler(feature_range=(0,1)).fit(x)
    x_scaled = scaler.transform(x)
    y = [x[0] for x in x_scaled]

    split_point = int(len(x_scaled)*0.8)

    x_train = x_scaled[:split_point]
    x_test = x_scaled[split_point:]
    y_train = y[:split_point]
    y_test = y[split_point:]

    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)

    time_step = 3 # the time step for the LSTM model
    xtrain = []
    ytrain = []
    xtest = []
    ytest = []
    for i in range(time_step,len(x_train)):
        xtrain.append(x_train[i-time_step:i,:x_train.shape[1]]) # we want to use the last 3 days’ data to predict the next day
        ytrain.append(y_train[i])
    for i in range(time_step, len(y_test)):
        xtest.append(x_test[i-time_step:i,:x_test.shape[1]])
        ytest.append(y_test[i])
    
    xtrain, ytrain = np.array(xtrain), np.array(ytrain)
    xtrain = np.reshape(xtrain,(xtrain.shape[0],xtrain.shape[1],xtrain.shape[2]))
    xtest, ytest = np.array(xtest), np.array(ytest)
    xtest = np.reshape(xtest,(xtest.shape[0],xtest.shape[1],xtest.shape[2]))
    data = [y[0] for y in x[split_point:]] # generating the original btc price sequence
    print(data[:5])
    return xtrain, ytrain, xtest, ytest, scaler

def predict_second(xtrain, ytrain, xtest, ytest, scaler):
    print(xtrain.shape, xtest.shape)
    model = Sequential()
    model.add(LSTM(units = 10, activation = 'relu', input_shape = (xtrain.shape[1], xtrain.shape[2]), return_sequences=True))
    model.add(LSTM(units = 5, activation = 'relu', input_shape = (xtrain.shape[1], xtrain.shape[2]), return_sequences=True))
    model.add(LSTM(units = 5, activation = 'relu', return_sequences=False ))
    model.add(Dense(units = 5)) 
    model.compile(loss='mean_squared_error',optimizer='adam')
    model.fit(xtrain,ytrain,epochs=100,validation_data=(xtest,ytest),batch_size=16,verbose=1)

    train_predict = model.predict(xtrain)
    test_predict = model.predict(xtest)

    #train_predict = np.c_[train_predict,np.zeros(train_predict.shape)]
    #test_predict = np.c_[test_predict,np.zeros(test_predict.shape)]
    #print(test_predict.shape)
    #print(train_predict.shape)
    train=scaler.inverse_transform(train_predict)
    test=scaler.inverse_transform(test_predict)
    print(train[:5])
    print(test[:5])

def fill(stock_info):

    get_data = stock_info
    get_data['H-L'] = stock_info.apply(lambda x: x['High'] - x['Low'], axis=1)
    get_data['O-C'] = stock_info.apply(lambda x: x['Open'] - x['Close'], axis=1)
    get_data['SMA7'] = get_data['Close'].rolling(7).mean()
    get_data['SMA14'] = get_data['Close'].rolling(14).mean()
    get_data['SMA21'] = get_data['Close'].rolling(21).mean()
    get_data['SMA50'] = get_data['Close'].rolling(50).mean()
    get_data['SMA100'] = get_data['Close'].rolling(100).mean()
    get_data['SMA200'] = get_data['Close'].rolling(200).mean()
    get_data['CMA30'] = get_data['Close'].expanding().mean()
    get_data['EWMA7'] = get_data['Close'].ewm(span=7).mean()
    get_data['EWMA14'] = get_data['Close'].ewm(span=14).mean()
    get_data['EWMA21'] = get_data['Close'].ewm(span=21).mean()
    get_data['EWMA100'] = get_data['Close'].ewm(span=100).mean()
    get_data['EWMA200'] = get_data['Close'].ewm(span=200).mean()
    #get_data['5d_future_close'] = get_data['Close'].shift(-5)
    #get_data['5d_close_future_pct'] = get_data['5d_future_close'].pct_change(5)
    get_data['Volume_1d_change'] = stock_info['Volume'].pct_change()
    get_data.dropna(inplace=True)

    return get_data

def find_mean(get_data):

    get_data = fill(get_data)
    close=get_data['Close']
    get_data.drop('Close', axis=1, inplace=True) 
    open=get_data['Open']
    get_data.drop('Open', axis=1, inplace=True) 
    high=get_data['High']
    get_data.drop('High', axis=1, inplace=True) 
    low=get_data['Low']
    get_data.drop('Low', axis=1, inplace=True) 
    h_L=get_data['H-L']
    get_data.drop('H-L', axis=1, inplace=True) 
    o_C=get_data['O-C']
    get_data.drop('O-C', axis=1, inplace=True) 
    volume=get_data['Volume']
    get_data.drop('Volume', axis=1, inplace=True) 
    volume_1d_change=get_data['Volume_1d_change']
    get_data.drop('Volume_1d_change', axis=1, inplace=True)
    print(list(get_data.columns))
    get_data['Mean']=get_data.mean(axis=1)
    print(get_data.tail())
    get_data['Close']=close
    get_data['Open']=close
    get_data['High']=close
    get_data['Low']=close
    get_data['H-L']=close
    get_data['O-C']=close
    get_data['Volume']=close
    get_data['Volume_1d_change']=close

    return get_data
    
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
    #get_data.drop('Dividends', axis=1, inplace=True) 
    #get_data.drop('Stock Splits', axis=1, inplace=True) 
    #get_data = fill(get_data)
    tech = applytechnicals(get_data)
    print(tech)
    #get_data = find_mean(get_data)
    #print(get_data['Mean'].tail())
    #xtrain, ytrain, xtest, ytest, scaler = process_test(get_data)
    #predict_second(xtrain,ytrain,xtest,ytest,scaler)
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


#######################################################
############### This is a new program #################
#######################################################
'''

SEQ_LEN = 60  # how long of a preceeding sequence to collect for RNN
FUTURE_PERIOD_PREDICT = 3  # how far into the future are we trying to predict?
RATIO_TO_PREDICT = "LTC-USD"
EPOCHS = 10  # how many passes through our data
BATCH_SIZE = 64  # how many batches? Try smaller batch if you're getting OOM (out of memory) errors.
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"


def classify(current, future):
    if float(future) > float(current):  # if the future price is higher than the current, that's a buy, or a 1
        return 1
    else:  # otherwise... it's a 0!
        return 0


def preprocess_df(df):
    df = df.drop("future", 1)  # don't need this anymore.

    for col in df.columns:  # go through all of the columns
        if col != "target":  # normalize all ... except for the target itself!
            df[col] = df[col].pct_change()  # pct change "normalizes" the different currencies (each crypto coin has vastly diff values, we're really more interested in the other coin's movements)
            df.dropna(inplace=True)  # remove the nas created by pct_change
            df[col] = preprocessing.scale(df[col].values)  # scale between 0 and 1.

    df.dropna(inplace=True)  # cleanup again... jic.


    sequential_data = []  # this is a list that will CONTAIN the sequences
    prev_days = deque(maxlen=SEQ_LEN)  # These will be our actual sequences. They are made with deque, which keeps the maximum length by popping out older values as new ones come in

    for i in df.values:  # iterate over the values
        prev_days.append([n for n in i[:-1]])  # store all but the target
        if len(prev_days) == SEQ_LEN:  # make sure we have 60 sequences!
            sequential_data.append([np.array(prev_days), i[-1]])  # append those bad boys!

    random.shuffle(sequential_data)  # shuffle for good measure.

    buys = []  # list that will store our buy sequences and targets
    sells = []  # list that will store our sell sequences and targets

    for seq, target in sequential_data:  # iterate over the sequential data
        if target == 0:  # if it's a "not buy"
            sells.append([seq, target])  # append to sells list
        elif target == 1:  # otherwise if the target is a 1...
            buys.append([seq, target])  # it's a buy!

    random.shuffle(buys)  # shuffle the buys
    random.shuffle(sells)  # shuffle the sells!

    lower = min(len(buys), len(sells))  # what's the shorter length?

    buys = buys[:lower]  # make sure both lists are only up to the shortest length.
    sells = sells[:lower]  # make sure both lists are only up to the shortest length.

    sequential_data = buys+sells  # add them together
    random.shuffle(sequential_data)  # another shuffle, so the model doesn't get confused with all 1 class then the other.

    X = []
    y = []

    for seq, target in sequential_data:  # going over our new sequential data
        X.append(seq)  # X is the sequences
        y.append(target)  # y is the targets/labels (buys vs sell/notbuy)

    return np.array(X), y  # return X and y...and make X a numpy array!


main_df = pd.DataFrame() # begin empty

ratios = ["BTC-USD", "LTC-USD", "BCH-USD", "ETH-USD"]  # the 4 ratios we want to consider
for ratio in ratios:  # begin iteration

    ratio = ratio.split('.csv')[0]  # split away the ticker from the file-name
    print(ratio)
    dataset = f'training_datas/{ratio}.csv'  # get the full path to the file.
    df = pd.read_csv(dataset, names=['time', 'low', 'high', 'open', 'close', 'volume'])  # read in specific file

    # rename volume and close to include the ticker so we can still which close/volume is which:
    df.rename(columns={"close": f"{ratio}_close", "volume": f"{ratio}_volume"}, inplace=True)

    df.set_index("time", inplace=True)  # set time as index so we can join them on this shared time
    df = df[[f"{ratio}_close", f"{ratio}_volume"]]  # ignore the other columns besides price and volume

    if len(main_df)==0:  # if the dataframe is empty
        main_df = df  # then it's just the current df
    else:  # otherwise, join this data to the main one
        main_df = main_df.join(df)

main_df.fillna(method="ffill", inplace=True)  # if there are gaps in data, use previously known values
main_df.dropna(inplace=True)
#print(main_df.head())  # how did we do??

main_df['future'] = main_df[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_PERIOD_PREDICT)
main_df['target'] = list(map(classify, main_df[f'{RATIO_TO_PREDICT}_close'], main_df['future']))

main_df.dropna(inplace=True)

## here, split away some slice of the future data from the main main_df.
times = sorted(main_df.index.values)
last_5pct = sorted(main_df.index.values)[-int(0.05*len(times))]

validation_main_df = main_df[(main_df.index >= last_5pct)]
main_df = main_df[(main_df.index < last_5pct)]

train_x, train_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_main_df)

print(f"train data: {len(train_x)} validation: {len(validation_x)}")
print(f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")
print(f"VALIDATION Dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")

model = Sequential()
model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(CuDNNLSTM(128, return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(CuDNNLSTM(128))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))


opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

# Compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')) # saves only the best ones

# Train model
history = model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(validation_x, validation_y),
    callbacks=[tensorboard, checkpoint],
)

# Score model
score = model.evaluate(validation_x, validation_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# Save model
model.save("models/{}".format(NAME))
'''