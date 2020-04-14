# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 08:21:24 2019

@author: skiff

#https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/
#by Jason Brownlee on June 7, 2016 in Deep Learning

"""

# https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/

import numpy as np
import pandas
import os
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn import random_projection

# sonar is of Rd where d = 60

os.chdir("G:\My Drive\data")

# fix random seed for reproducibility
k = 5 # setting k parameter for flip probability
seed = 7
np.random.seed(seed)


# load dataset
dataframe = pandas.read_csv("sonar.csv", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:60].astype(float)
Y = dataset[:,60]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# constructing a custom loss (from Keras documentation)
def flip_probability(y_true,y_pred):
    model = Sequential()
    model.fit(X, encoded_Y)
    y_prob = estimator.predict(X)
    X_rand = np.random.rand(k,X.shape[1]) #random projection matrix
    #transformer = random_projection.GaussianRandomProjection()
    #X_new = transformer.fit_transform(X)
    R_x = np.dot(X_rand,X.transpose()) # R * x_l
    last_weights = estimator.model.layers[0].get_weights()[0] 
    R_lw = np.dot(last_weights.transpose(),X_rand.transpose())
    projected_signs = np.sign(np.dot(R_lw,R_x))
    original_signs = np.sign(np.dot(last_weights,X.transpose()))
    fp = np.sum(projected_signs == original_signs)/(projected_signs.shape[0]*projected_signs.shape[1])
    return fp

# baseline model
def create_baseline():
	# create model
	model = Sequential()
	model.add(Dense(60, input_dim=60, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss=flip_probability, optimizer='adam', metrics=['accuracy']) #flip_probability
	return model

# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
estimator.fit(X, encoded_Y)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

estimator.fit(X, encoded_Y)

#estimator.fit(X, encoded_Y)
#y_prob = estimator.predict(X)
#pred_y_classes = y_prob.argmax(axis=-1)
#print(y_prob)
#print(len(pred_y_classes))
#print(encoded_Y)

#np.dot(y_prob.transpose(),encoded_Y)
#y_prob.shape() # dimensions
#encoded_Y.shape()
#last_weights = estimator.model.layers[0].get_weights()[0]
#np.sign(np.dot(last_weights,X.transpose()))


