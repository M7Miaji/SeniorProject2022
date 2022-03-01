# This algorithm is designed to pick between the stocks 
'''
stockInfo = {
            'Company Name': 'AAPL',
            'Industry': 'Tech',
            'Opening Price': 23.4,
            'Closing Price': 19.6,
            'Current Price': 20.9,
            'Enterprise Value': 480000, 
            'Company Value': 270000,
            'Years of Operation': 3,
            'Profit to Earning': 16,
            'Return on Equity': 0.9, # Percentage 
            'Net Proft Margin': 1.1, # Percentage
            'Price to Book': 1.2,
            'Return on Capital': 0.9, # Percenatge
            'Volume Traded': 20000
}

1-3 --> High Risk
4-6 --> Mid Risk
7-9 --> Low Risk

min = 15
max = 25
'''

'''
The Algorithm works by giving each of the metrics a value,
which will than be added up to decide the level of the risk.
'''

'''from pylivetrader.api import order_target, symbol

def initialize(context):
    context.i = 0
    context.asset = symbol('AAPL')

def handle_data(context, data):
    short_mavg = data.history(context.asset, 'price', bar_count=100, frequency="1m").mean()
    long_mavg = data.history(context.asset, 'price', bar_count=300, frequency="1m").mean()

    if short_mavg > long_mavg:
        order_target(context.asset, 100)
    elif short_mavg < long_mavg:
        order_target(context.asset, 0)'''

def chooseAlg(dict):
    count = 0
    if dict['Current Price'] >= min and dict['Current Price'] <= max:
        if (dict['Opening Price'] - dict['Closing Price']) >= 3:
            count = count + 0
        else:
            count = count + 1

        if dict['Enterprise Value'] >=  dict['Company Value']:
            count = count + 0
        else:
            count = count + 1
        
        if dict['Years of Operation'] <= 5:
            count = count + 0
        else:
            count = count + 1
        
        if dict['Profit to Earning'] >= 25:
            count = count + 0
        else:
            count = count + 1

        if dict['Return on Equity'] <= 10:
            count = count + 0
        else:
            count = count + 1
        
        if dict['Net Profit Margin'] <= 10:
            count = count + 0
        else:
            count = count + 1
            
        if dict['Price to Book'] >= 1:
            count = count + 0
        else:
            count = count + 1
        
        if dict['Return on Capital'] <= 2:
            count = count + 0
        else:
            count = count + 1
        
        if dict['Volume Traded'] <= 50000:
            count = count + 0
        else:
            count = count + 1
    return count