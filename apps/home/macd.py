import numpy as np
import pandas as pd
import yfinance as yf 
import ta
def stock(stock_symbol):
    stock_info = yf.Ticker(stock_symbol)
    return stock_info

def history(stock_ticker, startDate):
	hist = stock_ticker.history(start=startDate)
	return hist

def EVM(df):
	df['EMA10'] = df.Close.ewm(span=10).mean()
	df['EMA30'] = df.Close.ewm(span=30).mean()
	df['MACD'] = df.EMA10 - df.EMA30
	df['signal'] = df.MACD.ewm(span=2).mean()	
	df['macd']=ta.trend.macd_diff(df['Close'])
	print('indicator added')
	return df

def main(stockName):
	startDate='2020-1-1'
	stock_info=history(stock(stockName), startDate)
	stock_info.drop('Dividends', axis=1, inplace=True)
	stock_info.drop('Stock Splits', axis=1, inplace=True)
	stock_info = EVM(stock_info)
	print(stock_info.tail())

	Buy = [0]
	Sell = [1]

	for i in range(2, len(stock_info)):
		if stock_info.MACD.iloc[i] > stock_info.signal.iloc[i] and stock_info.MACD[i-1] < stock_info.MACD[i-1]:
			Buy.append[i]
		if stock_info.MACD.iloc[i] < stock_info.signal.iloc[i] and stock_info.MACD[i-1] > stock_info.MACD[i-1]:
			Sell.append[i]
		if stock_info.macd.iloc[i] < 0:
			if Sell[0] != 1:
				Sell.append(1)

	signal = ""

	if Buy[-1] > Sell[-1]:
		signal = "Buy"
	elif Buy[-1] < Sell[-1]:
		signal = "Sell"

	return Buy, Sell, stock_info, signal
	