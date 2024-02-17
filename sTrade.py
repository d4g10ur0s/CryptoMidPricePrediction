from binance.client import Client
from auth.binance_auth import *
from binance.exceptions import BinanceAPIException

def getInfo():
    # Initialize Binance client
    client = load_binance_creds('auth/auth.yml')
    # Fetch margin account balance
    account_info = client.get_margin_account()
    # Filter assets for USDT and XRP
    assets_to_check = ['USDT', 'XRP']
    filtered_assets = [asset for asset in account_info['userAssets'] if asset['asset'] in assets_to_check]
    # Print balance for filtered assets
    for asset in filtered_assets:
        print(f"Asset: {asset['asset']}, Borrowed: {asset['borrowed']}, Free: {asset['free']}, Interest: {asset['interest']}, Locked: {asset['locked']}")

def sTrade(midPrice) :
    # Initialize Binance client
    client = load_binance_creds('auth/auth.yml')
    # Define trade parameters
    symbol = 'XRPUSDT'  # Trading pair
    side = 'SELL'         # Buy or sell
    quantity = 10        # Quantity of XRP to buy
    price = midPrice     # None for market order, specify price for limit order
    leverage = 3         # Leverage level (e.g., 3x leverage)
    # Place margin trade
    try :
        order = client.create_margin_order(
            symbol=symbol,
            side=side,
            type='MARKET',  # Market order type
            quantity=quantity,
            price=price,
            newOrderRespType='FULL',  # Receive full order response
            leverage=leverage  # Specify leverage level
        )
    except BinanceAPIException as e:
        print(f"An error occurred: {e}")
    finally :
        # Print order response
        print(order)
