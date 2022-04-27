from matplotlib.pyplot import get, hist
import yfinance as yf 
import pandas as pd
import numpy as np

def stock(stock_symbol):
    stock_info = yf.Ticker(stock_symbol)
    return stock_info

def history(stock_ticker):
	hist = stock_ticker.history(period="max")
	return hist

df = pd.DataFrame

#print(history(stock('MSFT')).head())
stock_info = history(stock('AAPL'))
#print(get_data[-10:])
#partData=get_data[['Close']].values
#print(partData[0:5])

print(stock_info.head())
get_data = stock_info
get_data = get_data['Close'].to_frame()
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
print(get_data.shape)
get_data.dropna(inplace=True)
print(get_data.shape)
print(get_data.tail())