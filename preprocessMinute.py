import pandas as pd
import os
'''
Columns of Interest :
                    1. Open price
                    2. Close price
                    3. High price
                    4. Low price
                    5. Trends
                    6. Mean price ?
'''
# Transform values in interval [0,1]
def preprocessFile(file_path,processed_file_path):
    df=pd.read_csv(file_path)
    # Normalize Volume
    df['Volume']=df['Volume']/df['Volume'].max()
    #df['Volume'] = pd.concat(pd.DataFrame(data=df.iloc[0]['Volume']),df.iloc[1:]['Volume']-df.iloc[:len(df)-1]['Volume'])
    temp=df.loc[0,'Volume']
    df['Volume'] = df['Volume'].diff()
    df.loc[0,'Volume'] = temp
    # Create a Mean price in interval column
    df['Mean price']=(df['High price']+df['Low price'])/2
    df["Trends"]=df['Mean price'].diff()
    df.loc[0,'Trends'] = df.loc[0,'Mean price']
    df['Time Trends'] = df['Close price']-df['Open price']
    #df['MA_8'] = df['Mean price'].rolling(window=8).mean()
    #df['MA_8'].fillna(df['Mean price'].mean(),inplace=True)
    # Drop unacesairy columns
    df = df.drop(columns=["Open time","Close time",
                          "Quote asset volume","Number of trades","Taker buy base asset volume",
                          "Taker buy quote asset volume","Ignore"])
    # Save the processed file
    df.to_csv(processed_file_path,index=False)
