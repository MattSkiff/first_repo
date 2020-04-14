""" Neural Network.

A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
implementation with TensorFlow. This example is using the MNIST database
of handwritten digits (http://yann.lecun.com/exdb/mnist/).

Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
from __future__ import print_function

import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("G://My Drive//data//mnist//", one_hot=True)

import numpy as np

#tf.reset_default_graph() 

# Parameters
learning_rate = 0.1
num_steps = 500
batch_size = 128
display_step = 100

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# input tensor for reference model prediction
#y_g = tf.placeholder("float",[None,num_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}


# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits)

k = 5
Rm = tf.convert_to_tensor(np.random.rand(k,X.shape[1]),dtype = 'float32') #random projection matrix (tf)

# Define custom loss
# https://stackoverflow.com/questions/35330117/how-can-i-run-a-loop-with-a-tensor-as-its-range-in-tensorflow


def fp_loss(X,Rm):
    hi = weights['h1']  #tf.nn.softmax(logits) # weights (h) of nn - maybe this is incorrect?
    hi_xj = tf.matmul(tf.transpose(hi), tf.transpose(X)) # dot product of classifier and data
    hi_rm = tf.matmul(tf.transpose(hi),tf.transpose(Rm)) # projected classifier (transpose classifier times random matrix transpose)
    Rm_xj = tf.matmul(Rm,tf.transpose(X))
    
    sign_hi_xj = tf.math.sign(hi_xj) # original signs of data times classifier
    sign_hi_rm_rm_xj = tf.math.sign(tf.matmul(hi_rm,Rm_xj,transpose_b = False)) # signs of projected classifier and data
    dim = tf.cast(tf.size(hi_xj),dtype = tf.float32)
    loss_fp = tf.math.divide(tf.math.reduce_sum(tf.cast(tf.math.equal(sign_hi_xj,sign_hi_rm_rm_xj),tf.float32)),dim)
    test = (tf.reduce_mean(hi)*0)
    return  test + loss_fp #loss_fp

#def fp_loss(hi,X,Rm):
#    hi = weights['h1']  #tf.nn.softmax(logits) # weights (h) of nn - maybe this is incorrect?
#    test1 = tf.reduce_mean(hi)
#    hi_xj = tf.matmul(tf.transpose(hi), tf.transpose(X)) # dot product of classifier and data
#    hi_rm = tf.matmul(hi,tf.transpose(Rm)) # projected classifier (transpose classifier times random matrix transpose)
#    Rm_xj = tf.matmul(Rm,X)
#    test = tf.reduce_mean(Rm_xj)
#    
##    hi_rm_Rm_xj = tf.matmul(hi_rm,Rm_xj)
##    signs = hi_rm_Rm_xj*hi_xj
##    signs = tf.maximum(signs,1)
##    signs = tf.nn.softmax(signs)
##    signs = tf.maximum(signs,1)
##    flips = tf.reduce_mean(signs)
##    #sign_hi_xj = tf.math.sign(hi_xj) # original signs of data times classifier
##    #sign_hi_rm_rm_xj = tf.math.sign(tf.matmul(hi_rm,Rm_xj,transpose_b = False)) # signs of projected classifier and data
##    dim = tf.cast(tf.size(hi_xj),dtype = tf.float32)
#    
#    return test1 + test#flips/dim #loss_fp
'''
Original Implementation
'''
## Define loss and optimizer
#loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)) #old
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
#train_op = optimizer.minimize(loss_op) # old
## Evaluate model
#correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
## Initialize the variables (i.e. assign their default value)
#init = tf.global_variables_initializer()
## Start training
#with tf.Session() as sess:
#    # Run the initializer
#    sess.run(init)
#    for step in range(1, num_steps+1):
#        batch_x, batch_y = mnist.train.next_batch(batch_size)
#        # Run optimization op (backprop)
#        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y}) # old
#        if step % display_step == 0 or step == 1:
#            # Calculate batch loss and accuracy
#            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
#            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
#            print("Step " + str(step) + ", Minibatch Loss= " + \
#                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
#                  "{:.3f}".format(acc))
#    print("Optimization Finished!")
'''
Flip probability implementation
'''    
# Define loss and optimizer
loss_fp = fp_loss(X = X,Rm = Rm)
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_fp = optimizer.minimize(loss_fp)
# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    for step in range(1, num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization fp (backprop)
        sess.run(train_fp, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            loss, acc = sess.run([loss_fp, accuracy], feed_dict={X: batch_x, Y: batch_y})
            loss, acc = sess.run([loss_fp, accuracy], feed_dict={X: batch_x, Y: batch_y})
            # Calculate batch loss and accuracy
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
    print("Optimization Finished!")    
    
    #dir(tf) # debugging "no tranpose attribute error"

    # Calculate accuracy for MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images,
                                      Y: mnist.test.labels}))
    
writer = tf.summary.FileWriter('./neural_network_raw_tf', sess.graph)

