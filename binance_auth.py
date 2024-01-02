from binance.client import Client
from binance.exceptions import BinanceAPIException
def load_binance_creds(file):
    auth = {
        
    }

    return Client(api_key = auth['binance_api'], api_secret = auth['binance_secret'])
