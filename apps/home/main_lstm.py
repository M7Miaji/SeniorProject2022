from turtle import forward, st
from typing import Type
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

np.set_printoptions(suppress=True)

startDate='2020-1-1'

get_data=history(stock('AAPL'), startDate)

get_data['TradeDate']=get_data.index

ClosePrice=get_data[['Close']].values

df=pd.DataFrame()

data_columns=[
    'Open',
    'Close',
    'High',
    'Low'
]

print(type(get_data))
