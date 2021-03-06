# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 00:31:24 2019

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 16:13:50 2019
Script Template from Kaggle: https://www.kaggle.com/realshijjang/tensorflow-binary-classification-with-sigmoid/data
@author: user
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import numpy as np
import pandas as pd
import io
import requests
import math
from scipy import stats

tf.reset_default_graph()

def feature_normalize(dataset):
    mu = np.mean(dataset,axis=0)
    sigma = np.std(dataset,axis=0)
    return (dataset - mu)/sigma

def str_to_int(df):
    str_columns = df.select_dtypes(['object']).columns
    print(str_columns)
    for col in str_columns:
        df[col] = df[col].astype('category')

    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    return df

def count_space_except_nan(x):
    if isinstance(x,str):
        return x.count(" ") + 1
    else :
        return 0

# https://stackoverflow.com/a/42523230
def one_hot(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode 
    @return a DataFrame with one-hot encoding
    """
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        del df[each]
        df = pd.concat([df, dummies], axis=1)
    return df

df_train = pd.read_csv('G:\\My Drive\\data\\exercises-logistic-regression-exercise-1\\data\\train.csv')
df_test = pd.read_csv('G:\\My Drive\\data\\exercises-logistic-regression-exercise-1\\data\\test.csv')

print (df_train.isnull().sum())

def pre_processing(df):
    df = str_to_int(df)
    stats.describe(df).variance
    return df

df_train = pre_processing(df_train)
df_test = pre_processing(df_test)

features = df_train.iloc[:,:-1].values # select all but last column
features = feature_normalize(features)
labels = df_train.iloc[:, -1].values # last column
labels.resize([len(labels),1])
print(features.shape, labels.shape)
stats.describe(features).variance

real_test_x = df_test.iloc[:,:-1].values
print(real_test_x.shape)

rnd_indices = np.random.rand(len(features)) < 0.80

train_x = features[rnd_indices]
train_y = labels[rnd_indices]
test_x = features[~rnd_indices]
test_y = labels[~rnd_indices]

feature_count = train_x.shape[1]
label_count = train_y.shape[1]
print(feature_count, label_count)

# inputs
training_epochs = 3000
learning_rate = 0.1
neurons = feature_count - 1
cost_history = np.empty(shape=[1],dtype=float)

X = tf.placeholder(tf.float32,[None,feature_count],name = "input_data")
Y = tf.placeholder(tf.float32,[None,label_count],name = "label_data")
is_training=tf.Variable(True,dtype=tf.bool)

# models
initializer = tf.contrib.layers.xavier_initializer()
# the author of this script appears to have misinterpreted the argument for the number of neurons as the number of hidden layers
h0 = tf.layers.dense(X, neurons, activation=tf.nn.relu, kernel_initializer=initializer,name = "input_layer")
# h0 = tf.nn.dropout(h0, 0.95)
h1 = tf.layers.dense(h0, label_count, activation=None,name = "out_layer")

weights = tf.get_default_graph().get_tensor_by_name(os.path.split(h0.name)[0] + '/kernel:0')

def fp_loss(X,weights,k):
    hi_xj = tf.matmul(X,weights,name = "hi_xj")
    # https://stackoverflow.com/questions/43917456/matrix-norm-in-tensorflow
    norm_hi_xj = tf.math.multiply(tf.sqrt(tf.reduce_sum(tf.multiply(weights,weights))),tf.sqrt(tf.reduce_sum(tf.multiply(X,X))),name = "norm_hi_xj") #,name = "norm_hi_xj")
    loss_fp = -0.5*k*tf.reduce_sum(tf.math.square(hi_xj/norm_hi_xj))
    return  loss_fp #loss_fp

#cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=h1)
cost = fp_loss(X,weights,5)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# prediction = tf.argmax(h0, 1)
# correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

predicted = tf.nn.sigmoid(h1)
correct_pred = tf.equal(tf.round(predicted), Y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(training_epochs + 1):
        sess.run(optimizer, feed_dict={X: train_x, Y: train_y})
        loss, _, acc = sess.run([cost, optimizer, accuracy], feed_dict={
                                 X: train_x, Y: train_y})
        cost_history = np.append(cost_history, acc)
        if step % 500 == 0:
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(
                step, loss, acc))
            
    # Test model and check accuracy
    print('Test Accuracy:', sess.run([accuracy, tf.round(predicted)], feed_dict={X: test_x, Y: test_y}))
        
print(cost_history.shape)
plt.plot(range(len(cost_history)),cost_history)
plt.axis([0,training_epochs,0,1])
plt.show()

tf.trainable_variables()
