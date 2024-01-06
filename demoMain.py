import os
import time
import datetime as dt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error
import tensorflow as tf
from keras.models import Sequential , load_model
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras import backend as K
# Custom modules
import datasetCreation as dC

# Global variables
symbol = 'XRPUSDT'
datasetName = '\\dataset.csv'
processedDatasetName = '\\processedDataset.csv'
modelName = '\\MoMoney.keras'
timeSeriesLength = 15

def build_model(input_shape):
    # create model
    model = Sequential()
    model.add(LSTM(input_shape[0],activation='sigmoid',recurrent_activation="tanh",
                   input_shape=input_shape, return_sequences=True))
    model.add(LSTM(input_shape[0]*input_shape[1],activation='tanh',recurrent_activation="sigmoid",return_sequences=True,))
    model.add(LSTM(input_shape[0],activation='sigmoid',recurrent_activation="tanh"))
    model.add(Dense(input_shape[1],activation='sigmoid'))
    model.add(Dense(2, activation='linear'))
    return model

def train_model(X_train, y_train, epochs, batch_size, learning_rate, momentum):
    input_shape = X_train.shape[1:]
    model = build_model(input_shape)
    # Compile the model
    optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    # Define early stopping
    early_stopping = EarlyStopping(monitor='loss', min_delta=5e-8, patience=1, verbose=1)
    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping], verbose=1)
    return model

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
        if not (len(lastPrediction)<5) :
            my_test = pd.DataFrame(data=a.loc[len(a)-1,["Mean price"]]).transpose()
            #mse = mean_absolute_error([lastPrediction[0].loc[0,"Mean price"]],my_test)
            mse = mean_squared_error([lastPrediction[0].loc[0,"Mean price"]],my_test)
            print('MSE : ' + str(mse))
        else:
            mse = 5e-6
            print('Cannot compute mse')
        # predict or train
        if (os.path.exists(os.getcwd()+modelName) and mse<5e-5):
            # make prediction
            df = (a.loc[int(a.shape[0] * 4/5):, :]).reset_index(drop=True)
            df['Mean price'] = df['Mean price'] / df['Mean price'].max()
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
            else:
                print('currentValue : '+str(currentValue) + ' < lastPrediction : ' + str(lastPrediction[len(lastPrediction)-1].loc[0,'Mean price']))
        elif timeThreshold==0:
            # make X and y
            df = (a.loc[int(a.shape[0] * 3/4):, :]).reset_index(drop=True)
            df_1 = df.loc[:, ['Mean price', 'Trends percentage']]
            print(np.isnan(df_1).any())
            df['Mean price'] = df['Mean price'] / df['Mean price'].max()
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
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=None, random_state=8, shuffle=True)
            print("NaN values in X_train:", np.isnan(X_train).any())
            print("NaN values in y_train:", np.isnan(y_train).any())
            # Set hyperparameters
            epochs = 25
            batch_size = int(timeSeriesLength/2)
            learning_rate = float(input('Set learning rate: '))
            momentum = float(input('Set momentum: '))
            # Train the model
            trained_model = train_model(X_train, y_train, epochs, batch_size, learning_rate, momentum)
            # Save the model
            trained_model.save(os.getcwd()+modelName)
        # get time again
        lastTimestamp = dt.datetime.timestamp(dt.datetime.now())
        time.sleep(10)
if __name__ == '__main__':
    main()
