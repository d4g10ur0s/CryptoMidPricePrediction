from binance.client import Client
from auth.binance_auth import *
from binance.exceptions import BinanceAPIException
from binance.enums import *
import datasetCreation as dC

def cancelOrder(client) :
    try :
        orders = client.get_open_margin_orders(symbol='XRPUSDT')
        # Filter open orders to include only sell orders
        open_orders = [order for order in open_orders if order['side'] == 'SELL']
        open_orders.sort(key=lambda x: x['time'])
        if open_orders:
            # Extract the oldest order
            oldest_order_id = open_orders[0]['orderId']
            # Cancel the oldest order
            cancel_response = client.cancel_order(symbol=symbol, orderId=oldest_order_id)
            print(f"Oldest order (ID: {oldest_order_id}) has been canceled.")
        else:
            print("No open orders found to cancel.")
    except BinanceAPIException as e:
        # Extract and print only the error code
        error_code = e.code
        print(f"An error occurred: {error_code}")

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
    symbol = 'XRPUSDT'      # Trading pair
    side = 'SELL'           # Buy or sell
    quantity = 25           # Quantity of XRP to buy
    price = midPrice        # None for market order, specify price for limit order
    leverage = 3            # Leverage level (e.g., 3x leverage)
    price = round(price, 4)
    # Place margin trade
    try :
        order = client.create_margin_order(
            symbol=symbol,
            side=SIDE_SELL,
            type=ORDER_TYPE_LIMIT,
            timeInForce=TIME_IN_FORCE_GTC,
            quantity=quantity,
            price=price
        )
        print("Type of order : " + str(order['side']) + "\nValue of order : " + str(order['price']))
    except BinanceAPIException as e:
        print(str(e.code))
        if e.code == -2010 :
            cancelOrder(client)
        print(f"An error occurred: {e}")
