# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:43:52 2019

@author: user
"""

import numpy as np
import pandas as pd
from keras import backend as kb
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# import data
dataset = pd.read_csv("G://My Drive//data//pima-indians-diabetes.data.csv")

# creating input features and target variables
X= dataset.iloc[:,0:8]
y= dataset.iloc[:,8].astype('float64')

#standardizing the input feature
sc = StandardScaler()
X = sc.fit_transform(X)
X 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

classifier = Sequential()
#First Hidden Layer
classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal', input_dim=8,name = 'fc1'))
#Second  Hidden Layer
classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal',name = 'fc2'))
#Output Layer
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal', name='output'))

k = 1
Rm = np.random.rand(k,X.shape[1]).astype('float64')
Rj = tf.convert_to_tensor(Rm,dtype = 'float64')

def flip_prob_wrapper(input_tensor,random_matrix):
    x = tf.convert_to_tensor(input_tensor,dtype = 'float64')
    R = random_matrix
    def flip_probability(y_true,y_pred):
#        layer_name = "output"
#        intermediate_layer_model = Model(inputs=classifier.input,outputs=classifier.get_layer(layer_name).output)
#        intermediate_output = intermediate_layer_model.predict(X)
        hi = tf.convert_to_tensor(classifier.layers[0].get_weights()[0],dtype = 'float64')
        hi_x = kb.dot(kb.transpose(hi),kb.transpose(x))
        hi_R = kb.dot(kb.transpose(hi),kb.transpose(R))
        R_x = kb.dot(Rj,kb.transpose(x)) # randomly projecting down data
        hi_R_R_x = kb.dot(hi_R,R_x)
        projected_signs = tf.convert_to_tensor(kb.sign(hi_R_R_x),dtype = 'float64')
        original_signs =  tf.convert_to_tensor(kb.sign(hi_x),dtype = 'float64')
        flips = kb.sum(tf.cast(kb.equal(projected_signs,original_signs),tf.float64))
        size = kb.cast((kb.int_shape(projected_signs)[0]*kb.int_shape(projected_signs)[1]),dtype = 'float64')
        return  (flips/size)
    return flip_probability



#def customLoss(y_true,y_pred):
#    return K.backend.sum(K.backend.log(y_true) - K.backend.log(y_pred))

#Compiling the neural network
classifier.compile(optimizer ='adam',loss=flip_prob_wrapper(X,Rj), metrics =['accuracy']) #input_tensor = X,random_matrix = Rj

#Fitting the data to the training dataset
classifier.fit(X_train,y_train, batch_size=10, epochs=100)

eval_model=classifier.evaluate(X_train, y_train)

# loss and accuracy (from classifier.compile)
eval_model 