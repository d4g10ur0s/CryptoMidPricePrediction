# Custom Modules
from auth.binance_auth import *
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
import numpy as np
import seaborn as sns
import torch
import matplotlib.pyplot as plt

def categorize_move(trend):
    if trend < -0.001:
        return '0001'
    elif -0.001 <= trend <= 0:
        return '0010'
    elif 0 < trend < 0.001:
        return '0100'
    else:
        return '1000'

# preprocess dataset for coin
def preprocessDataset(dsName ,pdsName):
    df = pd.read_csv(dsName)
    # Calculate correlation matrix
    df = df.drop(columns=["Open time","Close time","Volume",'Taker buy base asset volume','Taker buy quote asset volume',
                          "Quote asset volume","Number of trades","Ignore",])
    # manual features
    # mean price
    df['Mean price']=(df['High price']+df['Low price'])/2
    # Mean averages seem to help
    df['MA_5'] = df['Mean price'].rolling(window=5).mean()
    df['MA_30'] = df['Mean price'].rolling(window=30).mean()
    df['MA_60'] = df['Mean price'].rolling(window=60).mean()
    #df['MA_9'] = df['Mean price'].rolling(window=9).mean()
    df['Log Earnings'] = np.log(df['Mean price']) - np.log(df['Mean price'].shift(-10))
    df['Percentage Earnings'] = (df['Close price'] - df['Open price'].shift(-10))/df['Open price'].shift(-10)
    df['Mean Price Earnings'] = (df['Mean price'] - df['Mean price'].shift(-10))/df['Mean price'].shift(-10)
    #df['Next price2'] = df['Mean price'].shift(-2)
    #df['Next price3'] = df['Mean price'].shift(-3)
    df = df.dropna()

    # let's see with fourrier
    '''
    for lag in range(5,20):
        # Compute autocorrelation matrix for all pairs of variables up to the specified lag
        for j in range(10) :
            interval_data = np.fft.fftn(df.iloc[j*lag : j*lag + lag])
            print(str(interval_data))
            dft_result = np.fft.fftn(interval_data)
            frequencies = np.fft.fftfreq(len(interval_data))
            magnitude_spectrum = np.abs(dft_result)
            print(str(frequencies))
            plt.figure(figsize=(10, 6))
            plt.plot(frequencies, magnitude_spectrum)
            plt.title(f'Magnitude Spectrum - {lag}')
            plt.xlabel('Frequency')
            plt.ylabel('Magnitude')
            plt.grid(True)
            plt.show()
            # something else
            print(str(np.log(np.abs(np.fft.fftshift(dft_result))**2)))
            plt.imshow(np.log(np.abs(np.fft.fftshift(dft_result))**2))
            plt.show()

    '''
    '''
    correlation_matrix = df.corr()
    # Create a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.show()

    # Calculate covariance matrix
    covariance_matrix = df.cov()
    # Create a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(covariance_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Covariance Heatmap')
    plt.show()

        # Plot autocorrelation matrix as a heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(autocorr_matrix, cmap='coolwarm', annot=True, fmt=".2f", cbar=True)
        plt.title(f'Autocorrelation Matrix at Lag {lag}')
        plt.xlabel('Variables')
        plt.ylabel('Variables')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        '''

    dfKeys = df.keys()
    for i in dfKeys :
        if i == 'Mean price' or i == 'Trends':
            pass
        else:
            df[i] = df[i]/df[i].abs().max()

    df = df.dropna()

    # Save the processed file
    df.to_csv(pdsName,index=False)
# create dataset for coin
def createDataset(pathName , symbol):
    tstartTime=dtime.datetime(2022,1,1,0,0,0)
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
    if len(df)>1051200:
        df[int(len(df)*0.2):].to_csv(dsName,index=False)
    else:
        df.to_csv(dsName,index=False)
    preprocessDataset(dsName ,pdsName)

def getCurrentValue(symbol):
    client = load_binance_creds('auth/auth.yml')
    ticker = client.get_ticker(symbol=symbol)
    print(f"The current value of {symbol} is {ticker['lastPrice']}")
    return ticker['lastPrice']
