"""
    A simple neural network written in Keras (TensorFlow backend) to classify the IRIS data
    https://gist.github.com/NiharG15/cd8272c9639941cf8f481a7c4478d525
    Nihar Gajare
    
    https://stackoverflow.com/questions/47619733/how-can-i-get-a-tensor-output-by-a-tensorflow-layer
    
"""

import numpy as np
import tensorflow as tf

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.optimizers import Adam


tf.logging.set_verbosity(tf.logging.ERROR)
tf.compat.v1.disable_eager_execution()
iris_data = load_iris() # load the iris dataset

print('Example data: ')
print(iris_data.data[:5])
print('Example labels: ')
print(iris_data.target[:5])

x = iris_data.data
y_ = iris_data.target.reshape(-1, 1) # Convert data to a single column

# One Hot encode the class labels
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y_)
#print(y)

# Split the data for training and testing
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.20)

# Build the model

model = Sequential()

model.add(Dense(10, input_shape=(4,), activation='relu', name='fc1'))
model.add(Dense(10, activation='relu', name='fc2'))
model.add(Dense(3, activation='softmax', name='output'))

layer_name = "output"
intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(x)

# https://stackoverflow.com/questions/46464549/keras-custom-loss-function-accessing-current-input-pattern
def custom_loss_wrapper(input_tensor):
        x_j = input_tensor
        def flip_probability(y_true,y_pred):
            k = 5
            layer_name = "output"
            intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
            intermediate_output = intermediate_layer_model.predict(x)
            R_m = np.random.rand(k,x.shape[1])
            h_i = model.layers[2].get_weights()[0]
            h_i_R_m = np.dot(h_i,R_m.transpose())
            R_m_x_j = np.dot(R_m,x_j)
            X_rand = np.random.rand(k,x.shape[1]) #random projection matrix
            #transformer = random_projection.GaussianRandomProjection()
            #X_new = transformer.fit_transform(X)
            R_x = np.dot(X_rand,X.transpose()) # R * x_l
            #last_weights = estimator.model.layers[0].get_weights()[0] 
            R_lw = np.dot(last_weights.transpose(),X_rand.transpose())
            projected_signs = np.sign(np.dot(R_lw,R_x))
            original_signs = np.sign(np.dot(last_weights,X.transpose()))
            return np.sum(projected_signs == original_signs)/(projected_signs.shape[0]*projected_signs.shape[1])
        return flip_probability

loss_fp = fp_loss(hi = weights['out'],X = X)
# Adam optimizer with learning rate of 0.001
optimizer = Adam(lr=0.001)
model.compile(optimizer, loss=custom_loss_wrapper(x), metrics=['accuracy'])

print('Neural Network Model Summary: ')
print(model.summary())

# Train the model
model.fit(train_x, train_y, verbose=2, batch_size=5, epochs=200)

# Test on unseen data

results = model.evaluate(test_x, test_y)

print('Final test set loss: {:4f}'.format(results[0]))
print('Final test set accuracy: {:4f}'.format(results[1]))

