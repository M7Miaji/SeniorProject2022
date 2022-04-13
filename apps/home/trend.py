from turtle import forward
from matplotlib.pyplot import get
import numpy as np #The Numpy numerical computing library
import pandas as pd #The Pandas data science library
import requests #The requests library for HTTP requests in Python
import xlsxwriter #The XlsxWriter libarary for 
import math #The Python math module
from scipy import stats #The SciPy stats module
#from yahoo_api import get_all, stocks
from IPython.display import display
from scraping import stock
from list import sample
from statistics import mean

stocker = 'AAPL,MSFT,GOOG,AMZN,TSLA,FB,NVDA,COST,ASML,INTC,CSCO,MU,OPEN,LCID,CSCO,RDBX,METC,TKNO,ZIMV,GOGO,AMD,GOOGL,NFLX'

symbol_strings = sample()

symbol_strings_test = stocker.split(",")

def portfolio_input():
    global portfolio_size
    portfolio_size = input("Enter the value of your portfolio:")

    try:
        val = float(portfolio_size)
    except ValueError:
        print("That's not a number! \n Try again:")
        portfolio_size = input("Enter the value of your portfolio:")

rv_columns = [
    'Ticker',
    'Price',
    'Number of Shares to Buy', 
    'Price-to-Earnings Ratio',
    'PE Percentile',
    'Price-to-Book Ratio',
    'PB Percentile',
    'Price-to-Sales Ratio',
    'PS Percentile',
    'EV/EBITDA',
    'EV/EBITDA Percentile',
    'PEG Ratio',
    'PEG Ratio Percentile',
    'RV Score'
]

rv_dataframe = pd.DataFrame(columns = rv_columns)
data = {}
count = 0
print("Checkpoint 1")
for symbol_string in symbol_strings_test:
    
    data = stock(symbol_string).info
    try:
        enterprise_value = data['enterpriseValue']
    except KeyError:
        enterprise_value = np.NaN

    try:
        ebitda = data['ebitda']
    except KeyError:
        ebitda = np.NaN

    try:
        ev_to_ebitda = enterprise_value/ebitda
    except TypeError:
        ev_to_ebitda = np.NaN

    try:
        forwardPE = data['forwardPE'] 
    except KeyError:
        forwardPE = np.NaN

    try:
        priceToBook = data['priceToBook'] 
    except KeyError:
        priceToBook = np.NaN

    try:
        priceToSales = data['priceToSalesTrailing12Months'] 
    except KeyError:
        priceToSales = np.NaN

    try:
        pegRatio = data['pegRatio'] 
    except KeyError:
        pegRatio = np.NaN

    try:
        regularMarketPrice = data['regularMarketPrice'] 
    except KeyError:
        regularMarketPrice = np.NaN

    rv_dataframe = rv_dataframe.append(
        pd.Series([
            symbol_string,
            regularMarketPrice,
            'N/A',
            forwardPE,
            'N/A',
            priceToBook,
            'N/A',
            priceToSales,
            'N/A',
            ev_to_ebitda,
            'N/A',
            pegRatio,
            'N/A',
            'N/A'
    ],
    index = rv_columns),
        ignore_index = True
    )

print("Checkpoint 2")
rv_dataframe[rv_dataframe.isnull().any(axis=1)]


for column in ['Price-to-Earnings Ratio', 'Price-to-Book Ratio','Price-to-Sales Ratio',  'EV/EBITDA','PEG Ratio']:
    rv_dataframe[column].fillna(rv_dataframe[column].mean(), inplace = True)

rv_dataframe[rv_dataframe.isnull().any(axis=1)]

metrics = {
            'Price-to-Earnings Ratio': 'PE Percentile',
            'Price-to-Book Ratio':'PB Percentile',
            'Price-to-Sales Ratio': 'PS Percentile',
            'EV/EBITDA':'EV/EBITDA Percentile',
            'PEG Ratio':'PEG Ratio Percentile'
}

for row in rv_dataframe.index:
    for metric in metrics.keys():
        rv_dataframe.loc[row, metrics[metric]] = stats.percentileofscore(rv_dataframe[metric], rv_dataframe.loc[row, metric])/100

# Print each percentile score to make sure it was calculated properly
for metric in metrics.values():
    print(rv_dataframe[metric])

#Print the entire DataFrame    
#rv_dataframe


for row in rv_dataframe.index:
    value_percentiles = []
    for metric in metrics.keys():
        value_percentiles.append(rv_dataframe.loc[row, metrics[metric]])
    rv_dataframe.loc[row, 'RV Score'] = mean(value_percentiles)
    
#rv_dataframe

rv_dataframe.sort_values(by = 'RV Score', inplace = True)
rv_dataframe = rv_dataframe[:50]
rv_dataframe.reset_index(drop = True, inplace = True)

portfolio_input()

position_size = float(portfolio_size) / len(rv_dataframe.index)
for i in range(0, len(rv_dataframe['Ticker'])-1):
    rv_dataframe.loc[i, 'Number of Shares to Buy'] = math.floor(position_size / rv_dataframe['Price'][i])
#rv_dataframe
display(rv_dataframe)
