# -*- coding: utf-8 -*-
"""
Created on Sun May 12 21:17:31 2019

@author: Jarvis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

####PRE-PROCESSING

#importing the training set
X = pd.read_csv('Google_Stock_Price_Train.csv')

#we need the date and open columns only : first two columns
training_set = X.iloc[:,1:2].values

#scaling - stardisation or normalisation x = x-min/max-min
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler( feature_range = (0,1) )
training_set_scaled = scaler.fit_transform(training_set)

#creating datastructure with 60 timesteps (60 PREVIOUS DAYS)
#and 1 output i.e the stock price

#x train will have past 60 days of memory
X_train =[]
y_train =[]   #furture stock price of next day

for i in range(60, 1258):  #starts from 60th entry as we need past memory
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
    
X_train, y_train = np.array(X_train), np.array(y_train)

#Reshaping - adding another dimension as RNN takes 3D input 
X_train = np.reshape(X_train, ( X_train.shape[0],
                               X_train.shape[1],
                               1 #no of feature =1 as only open 
                               )
                    )
##X_train = np.reshape(X_train, (1198,60,1))
    
####BUILDING THE RNN

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#initialising the NN
regressor = Sequential() #regressor as stock is continous o/p

#adding the first LSTM layer and DROPOUT  regularisation layer
regressor.add(LSTM(units = 50,
                   return_sequences = True, #this makes it return state/ forward to ahead layer 
                   input_shape = (X_train.shape[1],1)
                   )
                   )

regressor.add(Dropout(rate = 0.2)) #dropping 20 percent of the nuerons
# 0.2 * 50 =  10 nuerons will be dropped randomly on every iteration


#adding 2nd LSTM layer and dropout to prevent over-fitting
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(rate = 0.2))

#adding 3rd LSTM layer and dropout to prevent over-fitting
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(rate = 0.2))

#adding 4th LSTM layer and dropout to prevent over-fitting
regressor.add(LSTM(units = 50, return_sequences = False))
regressor.add(Dropout(rate = 0.2))


#adding the output layer - fully connected layer
regressor.add(Dense(units = 1))

#compiling the NN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error',metrics = ['accuracy'])

#fitting the RNN to the training set
regressor.fit(X_train, y_train, epochs = 10, batch_size = 32 )
#backpropagate/update weights every 32 stockprices i.e batchsize



#### MAKING PREDICTIONS AND VISULAISING THE RESULTS

#real stock price of Jan'17
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
test_set = dataset_test.iloc[:,1:2].values

#getting the predicted results of each day in Jan'17
#we need the previous 60 stock prices for this




  





