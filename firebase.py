
import requests

url = "https://yh-finance.p.rapidapi.com/market/get-popular-watchlists"

headers = {
    'x-rapidapi-host': "yh-finance.p.rapidapi.com",
    'x-rapidapi-key': "0c4fce0d32mshe17157162d6097dp1b5756jsn0958cc190c0c"
    }

response = requests.request("GET", url, headers=headers)

#store to jason file
with open('json_data.json', 'w') as outfile:
    outfile.write(response.text)
print('Done')
