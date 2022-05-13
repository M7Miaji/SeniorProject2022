import numpy as np
import pandas as pd
import yfinance as yf 

def stock(stock_symbol):
    stock_info = yf.Ticker(stock_symbol)
    return stock_info

def history(stock_ticker, startDate):
	hist = stock_ticker.history(start=startDate)
	return hist

def BBANDS(df, n):
    MA = df.Close.rolling(window=n).mean()
    SD = df.Close.rolling(window=n).std()
    df['UpperBB_'+str(n)] = MA + (2 * SD) 
    df['LowerBB_'+str(n)] = MA - (2 * SD)
    return df

def main_bbands(stockName):
	startDate='2020-1-1'
	stock_info=history(stock(stockName), startDate)
	stock_info['TradeDate']=stock_info.index
	stock_info.drop('Dividends', axis=1, inplace=True)
	stock_info.drop('Stock Splits', axis=1, inplace=True)
	print(stock_info)
	for i in range(2, 10, 2):
		stock_info = BBANDS(stock_info,i*10)
	stock_info.dropna(inplace=True)
	print(stock_info)

	Buy = []
	Sell = []
	open_pos = False

	for i in range(len(stock_info)):
		if stock_info.LowerBB_20[i] > stock_info.Close[i]:
			if open_pos == False:
				Buy.append(i)
				open_pos = True
		elif stock_info.UpperBB_20[i] < stock_info.Close[i]:
			if open_pos == False:
				Sell.append(i)
				open_pos = False
	signal = ""

	if Buy[-1] > Sell[-1]:
		signal = "Buy"
	else:
		signal = "Sell"

	return Buy, Sell, stock_info, signal

