# This algorithm is designed to pick between the stocks 

stockInfo = ['P/E', 'PEG', 'P/B', 'D/EBITDA', 'D/E', 'ROE', 'EV', 'EBT', 'OPM', 'AT', 'IER', 'EM', 'TRR', 'TR', 'NPM', 'CASH-FLOW', 'MARKET-PER', 'CPM']
import requests

url = "https://yh-finance.p.rapidapi.com/stock/v2/get-statistics"

querystring = {"symbol":"RMO","region":"US"}

headers = {
    'x-rapidapi-host': "yh-finance.p.rapidapi.com",
    'x-rapidapi-key': "b7e609f330msh515f9ad6685d6b9p101692jsn4be47c1196cc"
    }

response = requests.request("GET", url, headers=headers, params=querystring)

print(response.text)