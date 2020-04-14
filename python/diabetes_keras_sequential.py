# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:43:52 2019
https://www.youtube.com/watch?v=2U6Jl7oqRkM
https://medium.com/datadriveninvestor/building-neural-network-using-keras-for-classification-3a3656c726c1
"""
import tensorflow as tf
import keras

import math
import numpy as np
import pandas as pd
from keras import backend as kb
from keras import Sequential
from keras.layers import Dense
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.callbacks import TensorBoard

#tf.reset_default_graph() 
#kb.clear_session()

#logdir = "G://My Drive//dev_working_folder//python//diabetes_keras_sequential//" + datetime.now().strftime("%Y%m%d-%H%M%S")
#tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=0,write_graph=True, write_images=False)

tf.keras.backend.set_floatx('float32')

# import data
#dataset = pd.read_csv("G://My Drive//data//pima-indians-diabetes.data.csv")
dataset = pd.read_csv("https://raw.githubusercontent.com/mikeizbicki/datasets/master/csv/uci/pima-indians-diabetes.csv")

# creating input features and target variables
X= dataset.iloc[:,0:8]
y= dataset.iloc[:,8]

#standardizing the input feature
sc = StandardScaler()
X = sc.fit_transform(X)
X 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X = X.astype(np.float32)
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

classifier = Sequential()
#First Hidden Layer
classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal', input_dim=8,name = 'fc1'))
#Second  Hidden Layer
classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal',name = 'fc2'))
#Output Layer
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal', name='output'))


k = 5
#Rm = kb.cast(np.random.rand(k,X.shape[1]),'float32')
#Rj = kb.cast(Rm,dtype = 'float32')

# https://stackoverflow.com/questions/49225211/keras-custom-loss-implementation-valueerror-an-operation-has-none-for-gradi
# https://math.stackexchange.com/questions/517482/approximating-a-maximum-function-by-a-differentiable-function
# otherwise not differentiable
#def greater_approx(x,y,error = 0.000001):
#    """
#    https://math.stackexchange.com/questions/517482/approximating-a-maximum-function-by-a-differentiable-function
#    using a differentiable approximation of modulus
#    """
#    greater = 0.5*(x+y+math.sqrt(((x-y)+error)**2))
#    return greater

#class NewCallback(Callback):
#    def __init__(self, current_epoch):
#        self.current_epoch = current_epoch
#
#    def on_epoch_end(self, epoch, logs={}):
#        K.set_value(self.current_epoch, epoch)

def flip_prob_wrapper(input_tensor):#,random_matrix):
    x = keras.backend.variable(input_tensor)
#    R = random_matrix
    def flip_probability(y_true,y_pred):
        hi = keras.backend.variable(classifier.layers[0].get_weights()[0])
        hi_x = kb.dot(kb.transpose(hi),kb.transpose(x))
        norm_hi_x  = tf.norm(hi)*tf.norm(x)
        fp = kb.pow(hi_x/norm_hi_x,2)
        return fp #fake_loss + flips/size
    return flip_probability

#classifier.layers[2].output

#def customLoss(y_true,y_pred):
#    return K.backend.sum(K.backend.log(y_true) - K.backend.log(y_pred))

#Instantiate TensorBoard
#tensorboard = TensorBoard(log_dir = "G://My Drive//dev_working_folder//python//diabetes_keras_sequential//".format(time()))

#Compiling the neural network
classifier.compile(optimizer ='adam',loss=flip_prob_wrapper(X_train), metrics =['accuracy']) #input_tensor = X,random_matrix = Rj

#Fitting the data to the training dataset
classifier.fit(X_train,y_train, batch_size=10, epochs=100) #,callbacks = [tensorboard_callback])

eval_model=classifier.evaluate(X_train, y_train)

# loss and accuracy (from classifier.compile)
eval_model 

#writer = tf.summary.FileWriter('./diabetes_keras_sequential', sess.graph)