# Binance Connection Modules
from binance.client import Client
# Operating System Modules
import os
# Time Modules
import time
import datetime as dtime
# Data Analysis Modules
import pandas as pd
# Custom Modules
import preprocessMinute as preproc

# Cryptocurrency Pair
symbol = 'XRPUSDT'
# Get in hourly interval
interval = None
# File
file_path = None
# A function for csv manipulation
def CSV_Manipulation(client,tstartTime=dtime.datetime(2018,5,4,0,0,0),df=None):
    global symbol
    global interval
    global file_path
    global multiplier
    i=0# Just an index for the loop
    # Get the current working directory
    while tstartTime<=dtime.datetime.now():
        print(str(tstartTime))
        startTimeMS=int(tstartTime.timestamp() * 1000)
        klines=client.get_klines(symbol=symbol,interval=interval,startTime=startTimeMS,limit=1000)
        if i==0 and len(df)<1 :
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
        i+=1
        # Debugging
        #if i==2:
        #    break
        if multiplier==1 or multiplier==15:
            tstartTime+=dtime.timedelta(minutes=1000*multiplier)
        else:
            tstartTime+=dtime.timedelta(hours=1000)
    #endwhile
    df=df.drop_duplicates(subset=['Open time'])
    df.reset_index(drop=True, inplace=True)
    # make a smaller dataset
    if len(df) > 525600*2 :
        df=df.iloc[int(len(df)-525600*3/2) : ].reset_index(drop=True)
    df.to_csv(file_path,index=False)
    return df
'''
***********
Main Script
***********
'''
def mainScript(fp,pfp,mul):
    global symbol
    global interval
    global file_path
    global multiplier
    # Set up authentication
    
    # Create a Binance Client
    client = Client(API_KEY,PRIVATE_KEY)
    multiplier = mul
    file_path = fp
    if mul == 1 :
        interval = Client.KLINE_INTERVAL_1MINUTE
    elif mul == 15 :
        interval = Client.KLINE_INTERVAL_15MINUTE
    else :
        interval = Client.KLINE_INTERVAL_1HOUR
    df = []# The Data
    if os.path.exists(file_path):
        pass
    else: # Create a csv containing data
        df=CSV_Manipulation(client,df=df)
    #endif
    # Read data from csv
    df=pd.read_csv(file_path)
    # Add the missing rows
    tstartTime=dtime.datetime.strptime(df.iloc[-1]["Open time"],"%Y-%m-%d %H:%M:%S")
    df=CSV_Manipulation(client,tstartTime=tstartTime,df=df)
    # Start Preprocessing
    preproc.preprocessFile(fp,pfp)
