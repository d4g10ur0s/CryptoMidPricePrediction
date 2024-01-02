# Custom Modules
from binance_auth import *
# Binance Modules
from binance.enums import *
from binance.client import Client
# Operating System Modules
import os
# Time Modules
import time
import datetime as dtime
# Data Analysis Modules
import pandas as pd
# preprocess dataset for coin
def preprocessDataset(dsName ,pdsName):
    df = pd.read_csv(dsName)
    df = df.drop(columns=["Open time","Close time",
                          "Quote asset volume","Number of trades","Taker buy base asset volume",
                          "Taker buy quote asset volume","Ignore"])
    # mean price
    df['Mean price']=(df['High price']+df['Low price'])/2
    # manual features
    # volume percentage
    df['Volume percentage'] = (df['Volume'].shift(periods=1) / df['Volume'] - 1)
    # time trend
    df['Time trend']=df['Close price']-df['Open price']
    # mean diff
    df["Trends"]=df['Mean price'].diff()
    df.loc[0,'Trends'] = df.loc[0,'Mean price']
    df["Trends"]=df["Trends"]/2
    # trends percentage
    df['Trends percentage'] = (df['Close price'] / df['Open price']) - 1
    # Save the processed file
    # normalize values
    df['Volume']=df['Volume']/df['Volume'].max()
    df['High price']=df['High price']/df['High price'].max()
    df['Low price']=df['Low price']/df['Low price'].max()
    df['Close price']=df['Close price']/df['Close price'].max()
    df['Open price']=df['Open price']/df['Open price'].max()
    df['Volume percentage']=df['Volume percentage']/df['Volume percentage'].max()
    df['Time trend']=df['Time trend']/df['Time trend'].abs().max()
    df['Trends']=df['Trends']/df['Trends'].max()
    df = df.dropna()

    df.to_csv(pdsName,index=False)
# create dataset for coin
def createDataset(pathName , symbol):
    tstartTime=dtime.datetime(2021,1,1,0,0,0)
    client = load_binance_creds('auth/auth.yml')
    interval = Client.KLINE_INTERVAL_1MINUTE
    df = pd.DataFrame(data=[])
    while tstartTime<=dtime.datetime.now():
        print(str(tstartTime))
        startTimeMS=int(tstartTime.timestamp() * 1000)
        klines=client.get_klines(symbol=symbol,interval=interval,startTime=startTimeMS,limit=1000)
        if len(df)<1 :
            df=pd.DataFrame(data=klines,columns=["Open time","Open price","High price",
                                                 "Low price","Close price","Volume","Close time",
                                                 "Quote asset volume","Number of trades","Taker buy base asset volume",
                                                 "Taker buy quote asset volume","Ignore"])
            # Set Up datetime to human-readable form
            df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
            df['Close time'] = pd.to_datetime(df['Close time'], unit='ms')
        else :
            t_df=pd.DataFrame(data=klines,columns=["Open time","Open price","High price",
                                                   "Low price","Close price","Volume","Close time",
                                                   "Quote asset volume","Number of trades","Taker buy base asset volume",
                                                   "Taker buy quote asset volume","Ignore"])

            # Set Up datetime to human-readable form
            t_df['Open time'] = pd.to_datetime(t_df['Open time'], unit='ms')
            t_df['Close time'] = pd.to_datetime(t_df['Close time'], unit='ms')
            if t_df.shape[0]>0 and t_df.shape[0]<1000:
                df=pd.concat([df,t_df])
                break
            elif t_df.shape[0]<1:
                pass
            else:
                df=pd.concat([df,t_df])
        tstartTime+=dtime.timedelta(minutes=1000)
    #endwhile
    df=df.drop_duplicates(subset=['Open time'])
    df.reset_index(drop=True, inplace=True)
    df.to_csv(pathName,index=False)
    return df
# update the dataset
def updateDataset(dsName , pdsName,symbol):
    client = load_binance_creds('auth/auth.yml')
    interval = Client.KLINE_INTERVAL_1MINUTE
    # Read data from csv
    df=pd.read_csv(dsName)
    # Add the missing rows
    tstartTime=dtime.datetime.strptime(df.iloc[-1]["Open time"],"%Y-%m-%d %H:%M:%S")
    while tstartTime<=dtime.datetime.now():
        print(str(tstartTime))
        startTimeMS=int(tstartTime.timestamp() * 1000)
        klines=client.get_klines(symbol=symbol,interval=interval,startTime=startTimeMS,limit=1000)
        t_df=pd.DataFrame(data=klines,columns=["Open time","Open price","High price",
        "Low price","Close price","Volume","Close time",
        "Quote asset volume","Number of trades","Taker buy base asset volume",
        "Taker buy quote asset volume","Ignore"])
        # Set Up datetime to human-readable form
        t_df['Open time'] = pd.to_datetime(t_df['Open time'], unit='ms')
        t_df['Close time'] = pd.to_datetime(t_df['Close time'], unit='ms')
        if t_df.shape[0]>0 and t_df.shape[0]<1000:
            df=pd.concat([df,t_df])
            break
        elif t_df.shape[0]<1:
            pass
        else:
            df=pd.concat([df,t_df])
        tstartTime+=dtime.timedelta(minutes=1000)
        #endwhile
    df=df.drop_duplicates(subset=['Open time'])
    df.reset_index(drop=True, inplace=True)
    df.to_csv(dsName,index=False)
    preprocessDataset(dsName ,pdsName)

def getCurrentValue(symbol):
    client = load_binance_creds('auth/auth.yml')
    ticker = client.get_ticker(symbol=symbol)
    print(f"The current value of {symbol} is {ticker['lastPrice']}")
    return ticker['lastPrice']
