import requests

url = "https://bb-finance.p.rapidapi.com/stock/get-statistics"

querystring = {"id":"aapl:us","template":"STOCK"}

headers = {
    'x-rapidapi-host': "bb-finance.p.rapidapi.com",
    'x-rapidapi-key': "b7e609f330msh515f9ad6685d6b9p101692jsn4be47c1196cc"
    }

response = requests.request("GET", url, headers=headers, params=querystring)

print(response.text)