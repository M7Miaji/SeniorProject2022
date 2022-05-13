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

def rsi(df):
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['price change'] = df['Close'].pct_change()
    df['Upmove'] = df['price change'].apply(lambda x: x if x > 0 else 0)
    df['Downmove'] = df['price change'].apply(lambda x: abs(x) if x < 0 else 0)
    df['avg Up'] = df['Upmove'].ewm(span=19).mean()
    df['avg Down'] = df['Downmove'].ewm(span=19).mean()
    df = df.dropna()
    df['RS'] = df['avg Up']/df['avg Down']
    df['RSI'] = df['RS'].apply(lambda x: 100-(100/x+1))
    df.loc[(df['Close'] > df['MA200']) & (df['RSI'] < 30), 'Buy'] = 'Yes'
    df.loc[(df['Close'] < df['MA200']) & (df['RSI'] < 30), 'Buy'] = 'No'
    return df

def getSignals(df):
    Buy = []
    Sell =[]

    for i in range(len(df)):
        if "Yes" in df['Buy'].iloc[i]:
            Buy.append(df.iloc[i+1].name)
            for j in range(1, 11):
                if df['RSI'].iloc[i + j] > 40:
                    Sell.append(df.iloc[i+j+1].name)
                    break
                elif j == 10:
                    Sell.append(df.iloc[i+j+1].name)
    return Buy, Sell

def main(stockName):
    startDate='2020-1-1'
    stock_info = history(stock(stockName), startDate)
    stock_info.drop('Dividends', axis=1, inplace=True)
    stock_info.drop('Stock Splits', axis=1, inplace=True)
    rsi_stock = rsi(stock_info)
    print(rsi_stock.tail())

    buy, sell = getSignals(rsi_stock)
    print(buy, sell)
    return rsi_stock
