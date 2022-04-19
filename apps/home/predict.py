from turtle import forward
from matplotlib.pyplot import get
import numpy as np #The Numpy numerical computing library
import pandas as pd #The Pandas data science library
import math #The Python math module
from scipy import stats #The SciPy stats module
#from yahoo_api import get_all, stocks
from IPython.display import display
from scraping import stock, history
from list import sample
from statistics import mean
from sklearn.svm import SVR
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

from sklearn.preprocessing import MinMaxScaler
#for deep learning model
from keras import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


stocker = 'AAPL'

df_pandas = history(stock(stocker))

df = pd.DataFrame()
#df_pandas['Date'] = df['Date'] 
df['Open'] = df_pandas['Open'] 
df['High'] = df_pandas['High'] 
df['Low'] = df_pandas['Low'] 
df['Last'] = df_pandas['Close'] 
df['Close'] = df_pandas['Close'] 
df['Total Traded Quantity'] = df_pandas['Volume'] 
df['Turnover(Lacs)'] = df_pandas['Stock Splits'] 
df = df[::-1]
df = df.reset_index(drop=True)

open_price = df.iloc[:,1:2]
train_set = open_price[:2000].values
test_set = open_price[2000:].values
print("Train size: ",train_set.shape)
print("Test size:",test_set.shape)

print("Checkpoint 1")
'''
dates = pd.to_datetime(df['Date'])
plt.plot_date(dates, open_price,fmt='-')
plt.savefig("test1final.png")
'''
sc = MinMaxScaler()
train_set_scaled = sc.fit_transform(train_set)

x_train = []
y_train = []
for i in range(60,2000):
    x_train.append(train_set_scaled[i-60:i,0])
    y_train.append(train_set_scaled[i,0])
x_train = np.array(x_train)
y_train = np.array(y_train)
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_train.shape
print("Checkpoint 2")
reg = Sequential()
reg.add(LSTM(units = 50,return_sequences=True,input_shape=(x_train.shape[1],1)))
reg.add(Dropout(0.2))
reg.add(LSTM(units = 50,return_sequences=True))
reg.add(Dropout(0.2))
reg.add(LSTM(units = 50,return_sequences=True))
reg.add(Dropout(0.2))
reg.add(LSTM(units=50))
reg.add(Dropout(0.2))
reg.add(Dense(units=1))
reg.compile(optimizer = 'adam',loss='mean_squared_error')
reg.fit(x_train,y_train, epochs=20, batch_size =1,verbose=2)
print("Checkpoint 3")
input = open_price[len(open_price)-len(test_set)-60:].values
input.shape
input = sc.transform(input)
print("Checkpoint 4")
x_test = []
for i in range(60,95):
    x_test.append(input[i-60:i,0])
x_test = np.array(x_test)
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
print("Checkpoint 5")
pred = reg.predict(x_test)
pred = sc.inverse_transform(pred)
plt.plot(test_set,color='green')
plt.plot(pred,color='red')
plt.title('Stock_prediction')
plt.show()




