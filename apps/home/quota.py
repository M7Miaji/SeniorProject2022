from unittest import result
from matplotlib.font_manager import json_dump
import requests
import json
url = "https://yh-finance.p.rapidapi.com/market/v2/get-quotes"

querystring = {"region":"US","symbols":"AAPL"}

headers = {
	"X-RapidAPI-Host": "yh-finance.p.rapidapi.com",
	"X-RapidAPI-Key": "b7e609f330msh515f9ad6685d6b9p101692jsn4be47c1196cc"
}

response = requests.request("GET", url, headers=headers, params=querystring)
response_data = json_loads(response.text)
#print(current['quoteResponse']['result']['0']['language'])
print("----------------------------------")
#print(response.text)