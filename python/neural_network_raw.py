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
import numpy as np
import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("G://My Drive//data//mnist//", one_hot=True)

#tf.reset_default_graph()  #debugging

# Parameters
learning_rate = 0.1
num_steps = 500
batch_size = 100
display_step = 100

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

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
#Rm = tf.convert_to_tensor(np.random.rand(k,X.shape[1]),dtype = 'float32') #random projection matrix (tf)
Rm = np.random.rand(k,X.shape[1]) #random projection matrix (tf)
Rm = tf.constant(Rm, tf.float32,shape = (k,X.shape[1]))

# Define custom loss
# https://stackoverflow.com/questions/35330117/how-can-i-run-a-loop-with-a-tensor-as-its-range-in-tensorflow

'''
Flip probability implementation
'''  
def fp_loss(X,Rm):
    hi = weights['h1']  #tf.nn.softmax(logits) # weights (h) of nn - maybe this is incorrect?
    hi_xj = tf.matmul(tf.transpose(hi), tf.transpose(X),name = "hi_xj") # dot product of classifier and data
    norm_hi_xj = tf.math.multiply(tf.norm(hi,axis = 0),tf.norm(X,axis = 0)) #,name = "norm_hi_xj")
    loss_fp = tf.reduce_sum(tf.math.square(hi_xj/norm_hi_xj))
    return  loss_fp #loss_fp
  
# Define loss and optimizer
loss_fp = fp_loss(X = X,Rm = Rm)
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
            print("Step " + str(step) + ", Minibatch Loss= " + format(loss) + ", Training Accuracy= " +  "{:.3f}".format(acc))

    print("Optimization Finished!")  

    # Calculate accuracy for MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images,
                                      Y: mnist.test.labels}))
    
writer = tf.summary.FileWriter('./neural_network_raw_tf', sess.graph) # logging for tensorboard

