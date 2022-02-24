# This algorithm is designed to pick between the stocks 
import chooseAlgorithm
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

max_user = 25
min_user = 20 
industry = 'Tech'
company = 'AAPL'
risk_level = 'High'

def user_check(dict, max, min, industry, company, risk_level):
    if company != dict['Company Name']:
        if industry != dict['Tech']:
            if dict['Current Price'] >= max and dict['Current Price'] <= min:
                # Check the risk
                risk = chooseAlgorithm.chooseAlg(dict)

                risk_t = ''

                if risk <=3:
                    risk_t = 'High'
                elif risk_t >=4 and risk<=6:
                    risk = 'Mid'
                else:
                    risk_t = 'Low'
                    
                if risk_level == risk_t:
                    return dict['Company Name'], dict['Current Price'], risk_t
                else:
                    return dict['Company Name'], dict['Current Price'], 'Null'
