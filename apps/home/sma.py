import numpy as np
import pandas as pd
import yfinance as yf 

def stock(stock_symbol):
    stock_info = yf.Ticker(stock_symbol)
    return stock_info

def history(stock_ticker, startDate):
	hist = stock_ticker.history(start=startDate)
	return hist

def SMA(df, ndays): 
    SMA = pd.Series(df['Close'].rolling(ndays).mean(), name = 'SMA_'+str(ndays)) 
    df = df.join(SMA) 
    return df

def main_sma(stockName):
	startDate='2020-1-1'
	stock_info=history(stock(stockName), startDate)
	stock_info['TradeDate']=stock_info.index
	stock_info.drop('Dividends', axis=1, inplace=True)
	stock_info.drop('Stock Splits', axis=1, inplace=True)
	print(stock_info)
	for i in range(2, 10, 2):
		stock_info = SMA(stock_info,i*10)
	stock_info.dropna(inplace=True)
	print(stock_info)
	print(stock_info.iloc[494])
	Buy = []
	Sell = []

	for i in range(len(stock_info)):
		if stock_info.SMA_20.iloc[i] > stock_info.SMA_80.iloc[i] and stock_info.SMA_20.iloc[i-1] < stock_info.SMA_80.iloc[i-1]:
			Buy.append(i)
		elif stock_info.SMA_20.iloc[i] < stock_info.SMA_80.iloc[i] and stock_info.SMA_20.iloc[i-1] > stock_info.SMA_80.iloc[i-1]:
			Sell.append(i)
	signal = ""

	if Buy[-1] > Sell[-1]:
		signal = "Buy"
	else:
		signal = "Sell"
		
	return Buy, Sell, stock_info, signal
