import pandas as pd
import requests

url = "https://bb-finance.p.rapidapi.com/stock/get-financials"

querystring = {"id":"aapl:us"}

headers = {
	"X-RapidAPI-Host": "bb-finance.p.rapidapi.com",
	"X-RapidAPI-Key": "b7e609f330msh515f9ad6685d6b9p101692jsn4be47c1196cc"
}

response = requests.request("GET", url, headers=headers, params=querystring)

print(response["result"]["name"])

print(response.text)