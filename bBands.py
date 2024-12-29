import os
import requests
import time
import datetime as dt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error

import tensorflow as tf
import keras as kr
from keras.models import Sequential , load_model
from keras.layers import LSTM, Dense, Dropout , SimpleRNN , Bidirectional , GRU, Input
from tensorflow.keras import regularizers
from keras.optimizers import SGD , Adagrad , RMSprop
from keras.callbacks import EarlyStopping
from keras import backend as K
import matplotlib.pyplot as plt
# Custom modules
import datasetCreation as dC
import bTrade as bT
import sTrade as sT
import keras.backend as K

# Global variables
symbol = 'XRPUSDT'
#datasetName = '\\dataset_eth.csv'
#processedDatasetName = '\\processedDataset_eth.csv'
datasetName = '\\dataset_1.csv'
processedDatasetName = '\\processedDataset_1.csv'
modelName = '/orig.keras'
timeSeriesLength = 55

def demoMain():
    '''
    Algorithm goes like this...:
    1. hold :
            balance upper and lower bands
            if buy -> lowerWindow*=5
            if sell -> upperWindow*=5
            if hold -> reset
    2. buy :
            decrease lower band
            lowerWindow*=2
            lowerStd-= lowerStd - sub^#buy
    3. sell :
            decrease upper band
            upperWindow*=5
            upperStd-= upperStd - sub^#sell
    '''
    global datasetName, processedDatasetName, modelName, timeSeriesLength
    #df = dC.createDataset(os.getcwd()+datasetName, symbol)
    # preprocess dataset
    #dC.preprocessDatasetBB(os.getcwd()+datasetName, os.getcwd()+processedDatasetName)
    # set last signal
    lastSignal=0
    # set windows
    window=upperWindow=lowerWindow=8
    # set std
    std_dev=lower_std_dev=upper_std_dev=3
    # set booleans hold - buy - sell
    mbool = [True , False , False]
    # set pows
    holdPow = buyPow = sellPow = 0
    sub=.125
    # update dataset
    dC.updateDataset(os.getcwd()+datasetName, os.getcwd()+processedDatasetName, symbol)
    dataDF=pd.read_csv(os.getcwd()+processedDatasetName)
    lastTimestamp = dt.datetime.timestamp(dt.datetime.now())
    timeThreshold = 0
    while 1:
        # Load processed dataset
        timeThreshold+=abs(lastTimestamp - dt.datetime.timestamp(dt.datetime.now()))
        if timeThreshold>60:
            newData = True
            timeThreshold = 0
            lastTimestamp = dt.datetime.timestamp(dt.datetime.now())
            # update dataset
            dC.updateDataset(os.getcwd()+datasetName, os.getcwd()+processedDatasetName, symbol)
            dataDF=pd.read_csv(os.getcwd()+processedDatasetName)
        data=pd.DataFrame(data=dataDF['Close price'].values.tolist(),columns=['Close price'])
        # add current value
        currentValue=float(dC.getCurrentValue(symbol))
        new_row=pd.DataFrame(data=[currentValue],columns=['Close price'])
        # create data
        data = pd.concat([data, new_row], ignore_index=True)
        if mbool[0]:# if hold
            # reset
            buyPow=sellPow=0
            # use default window
            data['MiddleBand']=data['Close price'].rolling(window=window).mean()
            data['StdDev']=data['Close price'].rolling(window=window).std()
            # set hold pow
            upper_std_dev=upper_std_dev-sub**holdPow
            lower_std_dev=lower_std_dev-sub**holdPow
            holdPow+=1
            data['UpperBand'] = data['MiddleBand'] + (upper_std_dev * data['StdDev'])
            data['LowerBand'] = data['MiddleBand'] - (lower_std_dev * data['StdDev'])
        elif mbool[1]:# if buy
            # reset pows
            holdPow=sellPow=0
            # set lower window
            lowerWindow*=2
            upperWindow=window
            upper_std_dev=std_dev-1
            lower_std_dev=lower_std_dev-sub**buyPow
            buyPow+=1
            # use different windows
            data['MiddleBand']=data['Close price'].rolling(window=lowerWindow).mean()
            data['StdDev']=data['Close price'].rolling(window=lowerWindow).std()
            data['LowerBand'] = data['MiddleBand'] - (lower_std_dev * data['StdDev'])
            data['MiddleBand']=data['Close price'].rolling(window=window).mean()
            data['StdDev']=data['Close price'].rolling(window=window).std()
            data['UpperBand'] = data['MiddleBand'] + (upper_std_dev * data['StdDev'])
        else:
            # reset pows
            holdPow=buyPow=0
            # set lower window
            upperWindow*=2
            lowerWindow=window
            lower_std_dev=std_dev-1
            upper_std_dev=upper_std_dev-sub**sellPow
            sellPow+=1
            # use different windows
            data['MiddleBand']=data['Close price'].rolling(window=upperWindow).mean()
            data['StdDev']=data['Close price'].rolling(window=upperWindow).std()
            data['UpperBand'] = data['MiddleBand'] + (upper_std_dev * data['StdDev'])
            data['MiddleBand']=data['Close price'].rolling(window=window).mean()
            data['StdDev']=data['Close price'].rolling(window=window).std()
            data['LowerBand'] = data['MiddleBand'] - (lower_std_dev * data['StdDev'])
        #endif
        data['Signal'] = 0.0
        data.loc[data['Close price'] > data['UpperBand'], 'Signal'] = 1.0  # Buy signal
        data.loc[data['Close price'] < data['LowerBand'], 'Signal'] = -1.0  # Sell signal
        results = data.iloc[len(data)-180:]
        # Plot results
        plt.figure(figsize=(15, 8))
        plt.plot(results['Close price'], label='Close Price', alpha=0.5)
        plt.plot(results['MiddleBand'], label='Middle Band')
        plt.plot(results['UpperBand'], label='Upper Band')
        plt.plot(results['LowerBand'], label='Lower Band')
        plt.scatter(results.index[results['Signal'] == 1], results['Close price'][results['Signal'] == 1], marker='^', color='green', label='Buy')
        plt.scatter(results.index[results['Signal'] == -1], results['Close price'][results['Signal'] == -1], marker='v', color='red', label='Sell')
        plt.legend()
        plt.title("Bollinger Bands Strategy")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.savefig("BolingerBandsSignals.png")
        plt.clf()
        current_signal = data['Signal'].iloc[len(data)-10:].values.tolist()
        for i in range(1,4):
            print(f"Trading Signal: {current_signal[-i]} (Buy/Sell)")
            currentValue = float(dC.getCurrentValue(symbol))
            if current_signal[-i]<0:
                mbool[0]=mbool[1]=False
                mbool[2]=True
                print("sell")
                for k in range(4):
                    sT.sTrade(currentValue)
                break
            elif current_signal[-i]>0:
                mbool[0]=mbool[2]=False
                mbool[1]=True
                print("buy")
                for k in range(4):
                    bT.bTrade(currentValue)
                break
            else:
                print("hold")
        #endfor
        # if last signal is hold , then hold
        if current_signal[-1]==0:
            mbool[2]=mbool[1]=False
            mbool[0]=True
        #endif
        time.sleep(3)
    #endwhile

def main():
    demoMain()

if __name__ == '__main__':
    main()
