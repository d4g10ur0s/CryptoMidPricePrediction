import os
import time
import datetime as dt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error
import tensorflow as tf
import keras as kr
from keras.models import Sequential , load_model
from keras.layers import LSTM, Dense, Dropout , SimpleRNN
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras import backend as K
import matplotlib.pyplot as plt
# Custom modules
import datasetCreation as dC
import bTrade as bT
import sTrade as sT

# Global variables
symbol = 'XRPUSDT'
datasetName = '\\dataset.csv'
processedDatasetName = '\\processedDataset.csv'
modelName = '\\MoMoney.keras'
timeSeriesLength = 5

def build_model(input_shape):
    # create model
    model = Sequential()
    model.add(LSTM(input_shape[0],activation='sigmoid',recurrent_activation="tanh",
                   input_shape=input_shape, return_sequences=True))
    model.add(LSTM(input_shape[0]*input_shape[1],activation='sigmoid',recurrent_activation="tanh",return_sequences=True,))
    model.add(LSTM(input_shape[0],activation='sigmoid',recurrent_activation="tanh"))
    model.add(Dense(2, activation='linear'))
    return model

def train_model(X_train, y_train, X_val, y_val, epochs, batch_size, learning_rate, momentum):
    input_shape = X_train.shape[1:]
    model = build_model(input_shape)
    # Compile the model
    optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    # Define early stopping
    early_stopping = EarlyStopping(monitor='loss', min_delta=5e-4, patience=2, verbose=1)
    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=1)
    return model , history

def main():
    global datasetName, processedDatasetName, modelName, timeSeriesLength
    if not os.path.exists(os.getcwd()+datasetName):
        # create dataset
        df = dC.createDataset(os.getcwd()+datasetName, symbol)
        # preprocess dataset
        dC.preprocessDataset(os.getcwd()+datasetName, os.getcwd()+processedDatasetName)
    # update dataset
    dC.updateDataset(os.getcwd()+datasetName, os.getcwd()+processedDatasetName, symbol)
    lastTimestamp = dt.datetime.timestamp(dt.datetime.now())
    # time threshhold
    timeThreshold = 0
    lastPrediction = []
    mse = 0
    while 1 :
        # Load processed dataset
        a = pd.read_csv(os.getcwd()+processedDatasetName)
        timeThreshold+=abs(lastTimestamp - dt.datetime.timestamp(dt.datetime.now()))
        if timeThreshold>60:
            timeThreshold = 0
            dC.updateDataset(os.getcwd()+datasetName, os.getcwd()+processedDatasetName, symbol)
        # comute mse
        try :
            bT.getInfo(client)
        except :
            pass

        if not (len(lastPrediction)<5) :
            my_test = pd.DataFrame(data=a.loc[len(a)-1,["Mean price"]]).transpose()
            #mse = mean_absolute_error([lastPrediction[0].loc[0,"Mean price"]],my_test)
            mse = mean_squared_error([lastPrediction[0].loc[0,"Mean price"]],my_test)
            print('MSE : ' + str(mse))
        else:
            mse = 5e-6
            print('Cannot compute mse')
        # predict or train
        if False and (os.path.exists(os.getcwd()+modelName) and mse<5e-5):
            # make prediction
            df = (a.loc[int(a.shape[0] * 0.95):, :]).reset_index(drop=True)
            X_test = np.array(df.iloc[len(df)-timeSeriesLength:], dtype=np.float32)
            loaded_model = load_model(os.getcwd()+modelName)
            # prediction
            predictions=loaded_model.predict(X_test.reshape((-1,timeSeriesLength,len(a.keys()))))
            lastPrediction.append(pd.DataFrame(data=predictions,columns=["Mean price","Trends percentage"]))
            # show current situation
            if len(lastPrediction)>=6:# last prediction is next prediction
                lastPrediction.pop(0)
            print(lastPrediction[len(lastPrediction)-1])
            currentValue = dC.getCurrentValue(symbol)
            mpred = predictions[0]
            if float(currentValue) > mpred[0]  :
                print('currentValue : '+str(currentValue) + ' > lastPrediction : ' + str(lastPrediction[len(lastPrediction)-1].loc[0,'Mean price']))
                for i in range(5):
                    try :
                        sT.sTrade(mpred[0]-mpred[0]*mpred[1])
                    except :
                        print("Could not make the order")
            else:
                print('currentValue : '+str(currentValue) + ' < lastPrediction : ' + str(lastPrediction[len(lastPrediction)-1].loc[0,'Mean price']))
                for i in range(5):
                    try :
                        bT.bTrade(mpred[0]+mpred[0]*mpred[1])
                    except :
                        print("Could not make the order")
        elif True or timeThreshold==0:
            # make X and y
            df = (a.loc[int(a.shape[0] * 0.75):, :]).reset_index(drop=True)
            df_1 = df.loc[:, ['Mean price', 'Trends']]
            print(np.isnan(df_1).any())
            X, y = [], []
            for i in range(df.shape[0] - timeSeriesLength):
                try:
                    y.append(df_1.iloc[i+timeSeriesLength+5])
                    X.append(df.iloc[i:i+timeSeriesLength])
                except:
                    print("Going for training.")
            # Training dataset in numpy format
            X = np.array(X, dtype=np.float32)
            y = np.array(y, dtype=np.float32)
            # Split the data into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=8)
            print("NaN values in X_train:", np.isnan(X_train).any())
            print("NaN values in y_train:", np.isnan(y_train).any())
            # Set hyperparameters
            epochs = 25
            batch_size = int(timeSeriesLength/2)
            learning_rate = float(input('Set learning rate: '))
            momentum = float(input('Set momentum: '))
            # Train the model
            trained_model , history = train_model(X_train, y_train, X_val, y_val, epochs, batch_size, learning_rate, momentum)
            # Save the model
            trained_model.save(os.getcwd()+modelName)
            # Plot training loss
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Training and Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()
        # get time again
        lastTimestamp = dt.datetime.timestamp(dt.datetime.now())
        time.sleep(10)

if __name__ == '__main__':
    main()
