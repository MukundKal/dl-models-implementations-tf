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

#initialising the NN as a sequence of layers
classifier = Sequential()

#adding input and hidden layers
#relu for hidden layers and sigmoid for o/p layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11 ))

#add another hidden layer #not needed but still - just to go deep
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu' ))

#add the output layer  #softmax = sigmoid for more than 2 classes output
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

#compiling the NN - using stochastic gradient descent algo ADAM
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'] )


#fit/train the NN on the training Set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 10 )


#evaluating the NN model
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


#predicting using a sample value
new_prediction = classifier.predict(scaler.transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])))
new_prediction = (new_prediction > 0.5)
print(new_prediction[0][0])


#using k-fold cross validation, getting a better surity of accuracy
#using k-fold cross validation from sklearn in keras

 
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11 ))
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu' ))
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer , loss = 'binary_crossentropy', metrics=['accuracy'] )
    return classifier

#training a new NN classifier now using kfold multiple training sets now
classifier = KerasClassifier(build_fn = build_classifier)

#finding the best hyperParameters using GridSearch

parameters = { 'batch_size' : [25, 32],
               'nb_epoch' : [100, 500,],
               'optimizer' : ['adam', 'rmsprop',]
                }

grid_search = GridSearchCV(estimator = classifier, param_grid = parameters,
                           scoring = 'accuracy', cv = 10)

grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_








