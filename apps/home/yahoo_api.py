import requests

def get_statistics():
	url = "https://yh-finance.p.rapidapi.com/stock/v2/get-statistics"

	querystring = {"symbol":"AAPL","region":"US"}

	headers = {
	"X-RapidAPI-Host": "yh-finance.p.rapidapi.com",
	"X-RapidAPI-Key": "b7e609f330msh515f9ad6685d6b9p101692jsn4be47c1196cc"
	}

	response = requests.request("GET", url, headers=headers, params=querystring)
	return response.text

print(get_statistics())