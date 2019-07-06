# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:,3:-1].values
y = dataset.iloc[:, -1].values 

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#encode the gender variable
labelencoder1 = LabelEncoder()
X[:,2] = labelencoder1.fit_transform(X[:,2])

#encode the countries..first encode to numbers...then one-hot encode them
labelencoder2 = LabelEncoder()
X[:,1] = labelencoder1.fit_transform(X[:,1])
onehot = OneHotEncoder(categorical_features=[1])
X = onehot.fit_transform(X).toarray()

#remove one of the one-hot encodes feature columns - dummy variable trap
X=X[:,1:]

#splitting
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Building the ANN
import keras
from keras.models import Sequential #initialises the NN
from keras.layers import Dense #adds hidden layers
from keras.layers import Dropout #randomly disable some nuerons to prevent overfitting



#initialising the NN as a sequence of layers
NNclassifier = Sequential()

#adding input and hidden layers
#relu for hidden layers and sigmoid for o/p layer
NNclassifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11 ))

#add drouput layer - remove some nuerons every iteration
NNclassifier.add(Dropout(rate = 0.1))

#add another hidden layer #not needed but still - just to go deep
NNclassifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu' ))


#add drouput layer - remove some nuerons every iteration
NNclassifier.add(Dropout(rate = 0.1))

#add the output layer  #softmax = sigmoid for more than 2 classes output
NNclassifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

#compiling the NN - using stochastic gradient descent algo ADAM
NNclassifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'] )


#fit/train the NN on the training Set
NNclassifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 10 )


#evaluating the NN model
y_pred = NNclassifier.predict(X_test)
y_pred = (y_pred > 0.5)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)














