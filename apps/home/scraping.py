import yfinance as yf 

def stock(stock_symbol):
    stock_info = yf.Ticker(stock_symbol)
    return stock_info
