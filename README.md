
## 1. General Analysis

## Correlation

**Correlation** is a statistical measure that captures the **strength and direction** of the linear relationship between two variables. It's a **normalized** version of covariance, ranging from -1 to 1.

**Properties:**

- **Positive correlation (0 < r <= 1):** As one variable increases, the other tends to increase as well (strong positive correlation at r close to 1, weak positive correlation closer to 0).
- **Negative correlation (-1 <= r < 0):** As one variable increases, the other tends to decrease (strong negative correlation at r close to -1, weak negative correlation closer to 0).
- **Zero correlation (r = 0):** There's no linear relationship between the variables.
- Correlation is generally preferred over covariance because it's scale-independent, making it easier to interpret the strength of the relationship between variables measured in different units.

**Formula:**

```
corr(X, Y) = cov(X, Y) / (std(X) * std(Y))
```

----

## 1.1 Default Attributes

#### 1.1.1 Correlation

![image](https://github.com/user-attachments/assets/860c1f2f-46dc-4c94-a752-b0cea137cedf)

- **Linear Relationships 1** : Open price , High Price , Low price , Close Price , Mean price .
- **Linear Relationships 2** : Volume , Assets' Volumes , Number of Trades .
- **Idea** : Mix the attributes

## 1.2 Mixing with Volume per Number of Trades

![image](https://github.com/user-attachments/assets/ef590288-b45e-49ef-8b30-d4babf844769)

- Volume per number of Trades has close to 0 correlation with attributes that volume had 0 correlation before .
- Now volume has increased correlation with attributes that had 0 before .

- **Idea** : Mix with asset volume analogy , how much it is bought while it is sold .

## 1.3 Mixing with Asset Volume Analogy

![image](https://github.com/user-attachments/assets/eb143ff5-5f18-4685-9be5-db26df4bc995)


- The analogy seem to have a strong linear dependence with attributes that hadn't seem for volume to correlate .

- **Idea** : Multiply MVA with volume per number of trades .


## 1.4 Mixing More

![image](https://github.com/user-attachments/assets/6acbb9ad-5483-4868-88b1-7931d7e9e66e)


```
df["VAPT"] = (df['Taker buy base asset volume']-df['Taker buy quote asset volume'])/(2*df["Number of trades"])

df["VAPV"] = (df['Taker buy base asset volume']-df['Taker buy quote asset volume'])/(2*df["Volume"])
```

- The use of mixing volume and number of trades with buying and selling rate , makes attributes linearly dependent .

## 1.5 Correlation in Time

**10 mins time series**

![image](https://github.com/user-attachments/assets/e1ae72a9-253f-428a-a3f7-9201b9b317b5)


- There is a linear dependence between next mean price and all attributes .

**45 mins time series**

![image](https://github.com/user-attachments/assets/c02c5d9f-482f-439f-a0e3-f0419e53cecd)


**55 minutes time series**

![image](https://github.com/user-attachments/assets/81be0b15-49f8-456b-88c9-857890f02d8c)


**60 minutes time series**

![image](https://github.com/user-attachments/assets/c834cee6-a07b-4d99-8aa3-9dd18d7677c8)


**Results**

- 45 min length seems to be OK !


## 2. Attempts

## 2.1 Keras - TensorFlow

#### 2.1.1 Model that is Valid of Working for XRP

```
# create model
model = Sequential()
model.add(Input(shape=input_shape))

model.add(LSTM(input_shape[0],activation='tanh',return_sequences=True,return_state=False,recurrent_activation='sigmoid',go_backwards=False))

model.add(Bidirectional(LSTM(input_shape[1]**2,activation='tanh',return_sequences=True,return_state=False,recurrent_activation='sigmoid',go_backwards=False)))

model.add(LSTM(input_shape[0],activation='tanh',return_sequences=False,return_state=False,recurrent_activation='sigmoid',go_backwards=False))

model.add(Dense(1, activation='linear'))
```

**Architecture**

- 4 Layer Model
- 1st Layer has number of nodes equal to the length of Time Series .
- 2nd Layer is Bidirectional and has number of nodes equal to the multiplication of Time Series' length with number of features of each Time Series .
- 3rd Layer is as 1st .
- Output Layer has 1 node and linear activation function as activation function .
- Input and Hidden Layers have LSTM nodes with tanh as activation function and sigmoid as recurrent activation function .
- The model is trained using SGD as optimizer and MSE as loss function .

**Hyperparameters**

- Learning Rate equals 0.05 .
- Momentum equals 0.8 .
- Batch Size equals 10 .
- Time Series length equals 30 minutes . 

![image](https://github.com/user-attachments/assets/927fd8e8-521e-46a2-a9de-204e7dcb4a60)


**Comments**

- Now always working .

#### 2.1.2 Same model with MAE as loss function

```
model = Sequential()

model.add(Input(shape=input_shape))

model.add(LSTM(input_shape[0],activation='tanh',return_sequences=True,return_state=False,recurrent_activation='sigmoid',go_backwards=False))

model.add(LSTM(input_shape[0] * input_shape[1],activation='tanh',return_sequences=True,return_state=False,recurrent_activation='sigmoid',go_backwards=False))

model.add(LSTM(input_shape[0],activation='tanh',return_sequences=False,return_state=False,recurrent_activation='sigmoid',go_backwards=False))

model.add(Dense(1, activation='linear'))
```

- Learning Rate : 0.001
- Momentum : 0.7


**Comments**
