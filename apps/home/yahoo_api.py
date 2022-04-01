from unittest import result
from webbrowser import get
from matplotlib.font_manager import json_dump
import requests
import json
def get_quotes(stock_name):
	url = "https://yh-finance.p.rapidapi.com/market/v2/get-quotes"

	querystring = {"region":"US","symbols":stock_name}

	headers = {
		"X-RapidAPI-Host": "yh-finance.p.rapidapi.com",
		"X-RapidAPI-Key": "b7e609f330msh515f9ad6685d6b9p101692jsn4be47c1196cc"
	}

	response = requests.request("GET", url, headers=headers, params=querystring)
	response_data = json.loads(response.text)
	return response_data
#print(current['quoteResponse']['result']['0']['language'])
#print(get_quotes("NKE")['quoteResponse']['result'][0]['postMarketPrice'])
#response_data.type()
#print(response.text)