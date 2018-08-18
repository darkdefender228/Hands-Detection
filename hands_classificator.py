#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 20:31:29 2018

@author: romanilechko
"""
#importing packages and libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

#importing packages and libraries
def model():
    classifier = Sequential()
    
    classifier.add(Convolution2D(128, (5, 5), input_shape=(50, 50, 1), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(3, 3), dim_ordering="tf"))
    
    classifier.add(Convolution2D(64, (4, 4), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(3, 3), dim_ordering="tf"))
    
    classifier.add(Convolution2D(32, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))

    
    classifier.add(Flatten())
    
    classifier.add(Dense(128, activation='relu'))
    classifier.add(Dense(3, activation='softmax'))
    
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
    
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=False)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_set = train_datagen.flow_from_directory(
            'Train',
            target_size=(50, 50),
            batch_size=32,
            class_mode='categorical')
    
    test_set = test_datagen.flow_from_directory(
            'Test',
            target_size=(50, 50),
            batch_size=32,
            class_mode='categorical')
    
    classifier.fit_generator(epochs=7,validation_steps=10, 
            generator = train_set,
            validation_data=test_set)
    return classifier

model = model()



