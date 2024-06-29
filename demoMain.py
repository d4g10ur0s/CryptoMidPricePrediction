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

# Define custom Softplus loss function
def softplus_loss(y_true, y_pred):
    return K.log(1 + K.exp(y_pred))

def get_public_ip():
    response = requests.get('https://api.ipify.org')
    return response.text

# Global variables
symbol = 'XRPUSDT'
datasetName = '\\dataset.csv'
processedDatasetName = '\\processedDataset.csv'
modelName = '\\MoMoney.keras'
timeSeriesLength = 30

def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    is_small_error = tf.abs(error) <= delta
    squared_loss = 0.5 * tf.square(error)
    linear_loss = delta * (tf.abs(error) - 0.5 * delta)
    return tf.where(is_small_error, squared_loss, linear_loss)

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def torchModel(input_shape):
    pass

def build_model(input_shape):
    print(str(input_shape))
    # create model
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(input_shape[0], activation='tanh',
                   input_shape=input_shape, return_sequences=True,return_state=False,recurrent_activation='sigmoid',go_backwards=False
                   ))
    model.add(Bidirectional(LSTM(input_shape[1]**2, activation='tanh',
                   input_shape=input_shape, return_sequences=True, return_state=False,recurrent_activation='sigmoid',go_backwards=False
                   )))
    model.add(LSTM(input_shape[0], activation='tanh',
                   input_shape=input_shape, return_sequences=False,return_state=False,recurrent_activation='sigmoid',go_backwards=False
                   ))
    model.add(Dense(1, activation='linear'))
    return model

def train_model(X_train, y_train, X_val, y_val, epochs, batch_size, learning_rate, momentum):
    input_shape = X_train.shape[1:]
    model = build_model(input_shape)
    # Compile the model
    optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
    #optimizer = RMSprop(learning_rate=learning_rate)
    #model.compile(optimizer=optimizer, loss=root_mean_squared_error)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    # Define early stopping
    early_stopping = EarlyStopping(monitor='loss', min_delta=1e-5, patience=2, verbose=1)
    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=1)
    return model , history

def demoMain():
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
    newData = True
    dumb = pd.read_csv(os.getcwd()+datasetName)
    maxMean = 1#((dumb['High price']+dumb['Low price'])/2).max()
    mpred = None

    while 1 :
        # Load processed dataset
        timeThreshold+=abs(lastTimestamp - dt.datetime.timestamp(dt.datetime.now()))
        if timeThreshold>60:
            newData = True
            timeThreshold = 0
            lastTimestamp = dt.datetime.timestamp(dt.datetime.now())
            dC.updateDataset(os.getcwd()+datasetName, os.getcwd()+processedDatasetName, symbol)
        # comute mse
        try :
            bT.getInfo(client)
        except :
            pass

        a = pd.read_csv(os.getcwd()+processedDatasetName)
        if not (len(lastPrediction)<10) and newData :
            newData = False
            my_test = pd.DataFrame(data=a.loc[len(a)-10,["Mean price"]]*maxMean).transpose()
            # predictions variable acts like a queue
            mse = mean_absolute_error([lastPrediction[0].loc[0,"Mean price"]*maxMean],my_test)
            #mse = mean_squared_error([lastPrediction[0].loc[0,"Mean price"]],my_test)
            print('MSE : ' + str(mse))
        else:
            mse = 5e-6
            print('Cannot compute mse')
        # predict or train
        if (os.path.exists(os.getcwd()+modelName) and mse<40e-4):
            # make prediction
            if newData :
                df = (a.loc[int(a.shape[0] * 0.95):, :]).reset_index(drop=True)
                X_test = np.array(df.iloc[len(df)-timeSeriesLength:], dtype=np.float32)
                X_test = X_test.reshape((-1,timeSeriesLength,len(a.keys())))
                loaded_model = load_model(os.getcwd()+modelName)
                # prediction
                predictions=loaded_model.predict(np.log(np.abs(np.fft.fftshift(np.fft.fftn(X_test)))**2))
                lastPrediction.append(pd.DataFrame(data=predictions,columns=["Mean price"]))
                mpred = predictions[0]

            # show current situation
            if len(lastPrediction)>=11:# last prediction is next prediction
                lastPrediction.pop(0)
            print('Mean price : ' + str(lastPrediction[len(lastPrediction)-1].loc[0, 'Mean price']*maxMean))
            #print('Trends : ' + str(round(lastPrediction[len(lastPrediction)-1].loc[0 , 'Trends'] , 3))+'%')
            currentValue = dC.getCurrentValue(symbol)
            if float(currentValue)>maxMean:
                maxMean=float(currentValue)
            print("Current Difference : " + str(abs(float(currentValue)-mpred[0]*maxMean)))
            if float(currentValue) > mpred[0] *maxMean :
                print('currentValue : '+str(currentValue) + ' > lastPrediction : ' + str(mpred[0] *maxMean))
                for i in range(3):
                    try :
                        if (abs(float(currentValue)-mpred[0]*maxMean)>4.5e-3):
                            bT.bTrade(float(currentValue) - .5e-3)
                        elif(abs(float(currentValue)-mpred[0]*maxMean)<=4e-3 and abs(float(currentValue)-mpred[0]*maxMean)>=1.5e-3):
                            sT.sTrade(float(currentValue))
                    except :
                        print("Could not make the order")
            else:
                print('currentValue : '+str(currentValue) + ' < lastPrediction : ' + str(mpred[0] *maxMean))
                for i in range(3):
                    try :
                        if(abs(float(currentValue)-mpred[0]*maxMean)>3e-3):
                            sT.sTrade(float(currentValue) + .5e-3)
                        elif(abs(float(currentValue)-mpred[0]*maxMean)>1e-3):
                            bT.bTrade(float(currentValue))
                    except :
                        print("Could not make the order")
            # get time again
            #lastTimestamp = dt.datetime.timestamp(dt.datetime.now())
        elif timeThreshold==0:
            # get time again
            #lastTimestamp = dt.datetime.timestamp(dt.datetime.now())
            lastPrediction = []
            mse = 5e-6
            # make X and y
            df = (a.loc[int(a.shape[0] * 0.87):, :]).reset_index(drop=True)
            print(str(df.keys()))
            df_1 = df.loc[:, ['Mean price']]
            print(np.isnan(df_1).any())
            X, y = [], []
            for i in range(df.shape[0] - timeSeriesLength):
                try:
                    y.append(df_1.iloc[i+timeSeriesLength+10])
                    # how about taking the dft ?
                    X.append(np.log(np.abs(np.fft.fftshift(np.fft.fftn(df.iloc[i:i+timeSeriesLength])))**2))
                except:
                    print("Going for training.")
            # Training dataset in numpy format
            X = np.array(X, dtype=np.float32)
            y = np.array(y, dtype=np.float32)
            # Split the data into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05, random_state=8)
            print("NaN values in X_train:", np.isnan(X_train).any())
            print("NaN values in y_train:", np.isnan(y_train).any())
            # Set hyperparameters
            epochs = 20
            batch_size = int(timeSeriesLength/5)
            learning_rate = 0.025
            momentum = 0.785
            '''
            learning_rate = float(input('Set learning rate: '))
            momentum = float(input('Set momentum: '))
            '''
            # Train the model
            trained_model , history = train_model(X_train, y_train, X_val, y_val, epochs, batch_size, learning_rate, momentum)
            # Save the model
            trained_model.save(os.getcwd()+modelName)
            # Plot the signals
            y_pred = trained_model.predict(X_val)
            print("MSE : " + str(np.mean( (y_val - y_pred)**2 )) )
            # Plot real points as circles
            plt.plot(range(0,200), y_val[:200, 0], marker='o', label='Real Points')  # Circles for real points
            # Plot predicted points as crosses (x)
            plt.plot( range(0,200), y_pred[:200, 0], marker='x', label='Predicted Points')  # Crosses for predicted points
            plt.title('Real Points vs Predicted Points')
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.legend()
            plt.savefig(str(dt.datetime.timestamp(dt.datetime.now()))+".png")
            plt.clf()  # Clear the current figure
            '''
            # Plot training loss
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Training and Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()
            '''
        time.sleep(2)

def main():
    while 1 :
        try :
            demoMain()
        except :
            print("Error occured")
        time.sleep(60 * 2)
if __name__ == '__main__':
    main()
