from matplotlib.pyplot import hist
import yfinance as yf 

def stock(stock_symbol):
    stock_info = yf.Ticker(stock_symbol)
    return stock_info

def history(stock_ticker):
	hist = stock_ticker.history(period="max")
	return hist

#print(history(stock('MSFT')))

