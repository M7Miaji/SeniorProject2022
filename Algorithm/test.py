import requests

url = "https://stock-data-yahoo-finance-alternative.p.rapidapi.com/ws/screeners/v1/finance/screener/predefined/saved"

querystring = {"scrIds":"day_gainers","count":"25"}

headers = {
    'x-rapidapi-host': "stock-data-yahoo-finance-alternative.p.rapidapi.com",
    'x-rapidapi-key': "b7e609f330msh515f9ad6685d6b9p101692jsn4be47c1196cc"
    }

response = requests.request("GET", url, headers=headers, params=querystring)

print(response.text)