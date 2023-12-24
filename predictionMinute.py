# Datetime Modules
import datetime as dt
import time
# Operating System Modules
import os
# Data Analysis Modules
import pandas as pd
import numpy as np
# Preprocessing Modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# Neural Network Modules
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from keras.optimizers import SGD
# Custom Modules
import minuteAttemt as fa
from keras.saving import register_keras_serializable

@register_keras_serializable()
def pairwise_distance_loss(y_true, y_pred):
    # Reshape the predicted embeddings
    y_pred = K.l2_normalize(y_pred, axis=1)
    # Calculate pairwise dot product
    dot_product = K.dot(y_pred, K.transpose(y_pred))
    # Calculate pairwise squared distances
    squared_distances = 1.0 - dot_product
    # Ensure distances are non-negative
    squared_distances = K.maximum(squared_distances, 0.0)
    # Sum of pairwise squared distances
    sum_squared_distances = K.sum(squared_distances)
    # Average the distances
    loss = sum_squared_distances / (2.0 * K.cast(K.shape(y_pred)[0], 'float32'))
    return loss

# Check for GPU availability
if tf.test.gpu_device_name():
    print('GPU is available.')
else:
    print('GPU is NOT available.')

# get data
lastTimestamp = dt.datetime.timestamp(dt.datetime.now())
# model parameters
model_name="1_min_model"
mul = 1
timeSeriesLength = 10
first_layer = timeSeriesLength**2
# The Cryptocurrency Pair Dataset
file_path=os.getcwd()+"\\xrp_usdt_dataset_1_minute.csv"
processed_file_path=os.getcwd()+"\\xrp_usdt_processed_dataset_1_minute.csv"
# last prediction
lastPrediction = []
mse=0
# read data
a = pd.read_csv(processed_file_path)
while 1:
    # read file
    print("Process by 1 minute")
    print('Time Diff : ' + str(abs(lastTimestamp - dt.datetime.timestamp(dt.datetime.now()))))
    # to get new data
    if 60<abs(lastTimestamp - dt.datetime.timestamp(dt.datetime.now())) :
        print("Getting new data .")
        fa.mainScript(file_path,processed_file_path,mul)
        # read csv again
        a = pd.read_csv(processed_file_path)
    # get mse between last prediction and 5 min after
    if len(lastPrediction)==5 :
        my_test = pd.DataFrame(data=a.loc[len(a)-1,["High price","Low price" , "Mean price"]]).transpose()
        mse = mean_squared_error(lastPrediction[0],my_test)
        print('MSE : ' + str(mse))
    else:
        mse = 5e-6
        print('Cannot compute mse')
    lastTimestamp = dt.datetime.timestamp(dt.datetime.now())
    # decide whether to train or predict
    try :
        if mse>1e-5:
            # for training
            print("Training Time !")
            # preprocess data
            df=(a.loc[int(a.shape[0] * 1/2):,:]).reset_index(drop=True)
            # the prediction
            df_1=df[["High price","Low price" , "Mean price"]]
            # noramalize values
            df['Open price']=df['Open price']/df['Open price'].max()
            df['High price']=df['High price']/df['High price'].max()
            df['Low price']=df['Low price']/df['Low price'].max()
            df['Close price']=df['Close price']/df['Close price'].max()
            df['Mean price']=df['Mean price']/df['Mean price'].max()
            # train the model
            X, y = [], []
            for i in range(df.shape[0] - timeSeriesLength):
                try :
                    y.append(df_1.iloc[i+timeSeriesLength+5])
                    X.append(df.iloc[i:i+timeSeriesLength])
                except :
                    print("Going for training .")
            # training dataset in numpy format
            X = np.array(X, dtype=np.float32)
            y = np.array(y, dtype=np.float32)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42, shuffle=True)
            # create model
            model = Sequential()
            model.add(LSTM(first_layer,activation='sigmoid',recurrent_activation="tanh",
                           input_shape=(timeSeriesLength, len(df.keys())), return_sequences=True))
            #model.add(LSTM(len(df.keys())**2,activation='sigmoid',recurrent_activation="tanh",return_sequences=True,))
            model.add(LSTM(timeSeriesLength,activation='sigmoid',recurrent_activation="tanh",return_sequences=True,))
            model.add(LSTM(timeSeriesLength*len(df.keys()),activation='sigmoid',recurrent_activation="tanh"))
            model.add(Dense(len(df.keys()),activation='sigmoid'))
            model.add(Dense(3, activation='sigmoid'))
            # early stopping
            callback = tf.keras.callbacks.EarlyStopping(
                                            monitor="loss",
                                            min_delta=5e-5,
                                            patience=1,
                                            verbose=1,
                                            )
            # Compile the model
            sgd = SGD(learning_rate=float(input('Set learning rate : '))
                    , momentum=float(input('Set momentum : ')))# Specify learning rate and momentum
            model.compile(optimizer=sgd, loss='mean_squared_error')
            bs=int(timeSeriesLength/2)
            model.fit(X_train, y_train, epochs=25, batch_size=bs,callbacks=[callback])
            # end of training
            # save the model
            model.save('dagklaMinute.keras')
        else:
            # preprocess data
            df=(a.loc[int(a.shape[0] * 1/2):,:]).reset_index(drop=True)
            # noramalize values
            df['Open price']=df['Open price']/df['Open price'].max()
            df['High price']=df['High price']/df['High price'].max()
            df['Low price']=df['Low price']/df['Low price'].max()
            df['Close price']=df['Close price']/df['Close price'].max()
            df['Mean price']=df['Mean price']/df['Mean price'].max()
            X_test = np.array(df.iloc[len(df)-timeSeriesLength:], dtype=np.float32)
            loaded_model = load_model('dagklaMinute.keras')
            # prediction
            predictions=loaded_model.predict(X_test.reshape((-1,timeSeriesLength,len(a.keys()))))
            lastPrediction.append(pd.DataFrame(data=predictions,columns=["High price","Low price" , "Mean price"]))
            if len(lastPrediction)==6:# last prediction is next prediction
                lastPrediction.pop(0)
            print(lastPrediction[len(lastPrediction)-1])
            time.sleep(60)
        # end while
    except ValueError as e:
        print(str(e))
        input("Exit ?")
