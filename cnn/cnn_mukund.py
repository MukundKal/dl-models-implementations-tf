# -*- coding: utf-8 -*-
"""
Created on Sun May 12 17:33:53 2019

@author: Jarvis
"""
#preprocessing already done in folders
#building th CNN

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#initialising the network - classifier
classifier = Sequential()


#adding the first layer - CONV layer
classifier.add(Conv2D(32,(3,3),input_shape = (64,64,3), activation = 'relu'))


#adding the pooling layer - MAXPOOL layer
classifier.add(MaxPooling2D(pool_size = (2,2) ))


#adding another CONV and POOLING layer pair to inc acc.
classifier.add(Conv2D(32,(3,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2) ))


#flattening the pooled feature maps to input to ANN fully connected layers
classifier.add(Flatten())

#adding the fully connected layers - DENSE layer
classifier.add(Dense(units = 128, activation = 'relu' ))
classifier.add(Dense(units = 1, activation = 'sigmoid'))


#compiling the NN
classifier.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])



#augumentating more images to increase train_set size to reduce overfitting

'''from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
'''
#Fit the CNN finally to the training set

classifier.fit_generator(
        training_set,
        steps_per_epoch = 8000,
        epochs = 25,
        validation_data = test_set,
        validation_steps = 2
       )


import numpy as np
from keras.preprocessing import image
test_image1 = image.load_img('dataset/single_prediction/cat_or_dog1_isDog.jpg',target_size = (64, 64))
test_image1 = image.img_to_array(test_image1)
test_image1 = np.expand_dims(test_image1, axis = 0)



result = classifier.predict(test_image1)

#tells which class is 1 and which is 0
training_set.class_indices

if result[0][0] == 1:
    prediction = 'dog'
    
else:
    predicton = 'cat'
