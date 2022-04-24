from lib2to3.pgen2.token import OP
import numpy as np #The Numpy numerical computing library
import pandas as pd #The Pandas data science library
from lstm_copy import run
from datetime import datetime
from scraping import stock, history
# To remove the scientific notation from numpy arrays
np.set_printoptions(suppress=True)

startDate='2020-1-1'
endDate=datetime(2020, 10, 5)

# Getting the Data
get_data=history(stock('AAPL'), startDate)
#print(get_data.head())
#print(get_data.shape)

get_data['TradeDate']=get_data.index

# Placing the data in variables
#FullData=get_data[['Close']].values

LowPrice=get_data[['Low']].values
HighPrice=get_data[['High']].values
ClosePrice=get_data[['Close']].values 
OpenPrice=get_data[['Open']].values
Volume=get_data[['Volume']].values

df = pd.DataFrame()

#print(df.head())
#print('Original Prices')
 
#print(HighPrice[-10:])

accuracy_close, orignal_close, predicted_close = run(ClosePrice)
#accuracy_high, orignal_high, predicted_high = run(HighPrice)
#accuracy_low, orignal_low, predicted_low = run(LowPrice)
#accuracy_open, orignal_open, predicted_open = run(OpenPrice)
#accuracy_volume, orignal_volume, predicted_volume = run(Volume)

#accuracy, orignal, predicted = 0 

#print(accuracy_volume)
#print(orignal_volume)
#print(predicted_volume)

#print(type(orignal_close))


#df['Open'] = predicted_open.tolist()
df['Close'] = orignal_close.tolist()
#df['High'] = predicted_high.tolist()
#df['Low'] = predicted_low.tolist()

print(df.head())
print(orignal_close.shape)
print(orignal_close)