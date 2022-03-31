import requests

url = "https://alpha-vantage.p.rapidapi.com/query"

querystring = {"function":"GLOBAL_QUOTE","symbol":"AAPL","datatype":"json"}

headers = {
	"X-RapidAPI-Host": "alpha-vantage.p.rapidapi.com",
	"X-RapidAPI-Key": "b7e609f330msh515f9ad6685d6b9p101692jsn4be47c1196cc"
}

response = requests.request("GET", url, headers=headers, params=querystring)

print(response.text)