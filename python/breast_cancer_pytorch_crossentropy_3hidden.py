# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 00:07:25 2019
https://medium.com/@prudhvirajnitjsr/simple-classifier-using-pytorch-37fba175c25c
@author: pmaddala

Adapted to
* use FP loss
* use the breast cancer dataset from sklearn
* split test/train 
* print extra figures / metrics
"""

import sklearn.datasets # importing breast cancer dataset
import sklearn.model_selection # test train split
import torch # pytorch
import numpy as np 
import time # measuring training time
import random

np.random.seed()
X,y = sklearn.datasets.load_breast_cancer(return_X_y = True) # data features, data labels
rand = random.randint(1,9999)

import matplotlib.pyplot as plt

# TODO: code for visualising data here

# splitting data randomly into test/train
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.33, random_state=rand) 

# casting data to torch tensors
X_test =  torch.from_numpy(X_test).type(torch.FloatTensor)# test features
y_test =  torch.from_numpy(y_test).type(torch.LongTensor)# test labels

X_train = torch.from_numpy(X_train).type(torch.FloatTensor)# training features
y_train = torch.from_numpy(y_train).type(torch.LongTensor)# training labels

import torch.nn as nn
import torch.nn.functional as F

#https://discuss.pytorch.org/t/normalizing-a-tensor-column-wise/20764/3
# unused presently
def normalize(x):
    x_normed = x / x.max(0, keepdim=True)[0]
    return x_normed

# https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
# initialise weights - with a (popular) and complicated weight initialisation scheme I don't understand (yet)
# this particular scheme is recommended for use with relu activation functions
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal(m.weight)
        m.bias.data.fill_(0.01)      

#our class must extend nn.Module
class Net(nn.Module):
    
    def __init__(self):
        super(Net,self).__init__()
        #Our network consists of 4 layers. 1 input, 2 hidden and 1 output layer
        
        #This applies Linear transformation to input data. 
        self.fc1 = nn.Linear(in_features = 30,out_features = 15)
        self.fc2 = nn.Linear(in_features = 15,out_features = 5)

        #This applies linear transformation to produce output data
        self.fc3 = nn.Linear(5,2)
        
    #This must be implemented
    def forward(self,x):
        #Activation function is Relu.
        
        #Output of the first hidden layer
        x = self.fc1(x)
        x = F.relu(x) 
        
        #Returning INPUT to second (last-but-1) hidden layer, not output
        out = x
        
        #Output of the second hidden layer
        x = self.fc2(x)
        x = F.relu(x) 
        
        #Output layer
        x = self.fc3(x)
        return x, out
        
    #This function takes an input and predicts the class, (0 or 1)        
    def predict(self,x):
        #Apply softmax to output
        pred = F.softmax(self.forward(x)[0])
        ans = []
        for t in pred:
            if t[0]>t[1]:
                ans.append(0)
            else:
                ans.append(1)
        return torch.tensor(ans)
               
#Initialize the model and weights
model = Net()
model.apply(init_weights)
print(model)

#Define the optimizer
#optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
optimizer = torch.optim.SGD(model.parameters(),lr=0.001)

#custom loss function
def fp_loss(activations, last_but_one_classifier):
    hi = last_but_one_classifier # the weights (including bias term) from the last-but-1 hidden layer
    xj = activations # the output of the last-but-1 hidden layer, after going through relu activation
    hi_xj = torch.mm(hi,xj.t()) # the matrix multiplication 
    norm_hi = torch.norm(hi,dim = 1)
    norm_xj = torch.norm(xj,dim = 1)
    fp_loss = torch.sum(torch.pow((torch.div(hi_xj,torch.mm(torch.unsqueeze(norm_hi,1),torch.unsqueeze(norm_xj,1).t()))),2))
    return fp_loss

criterion = nn.CrossEntropyLoss()

#Number of epochs
epochs = 5000
#List to store losses / accuracy
losses = []
accuracy = []

start = time.time()

for i in range(epochs):
    #Predict the output for Given input
    y_pred = model.forward(X_train)
    
    #Get Last-but-1 layer INPUT on forward pass
    # Adding column of 1s to activations (bias)
    activations = torch.cat([model.forward(X_train)[1],torch.ones(len(X_train)).unsqueeze(1)],1) 
    
    #Get weights matrices and biases of network as a list
    layers=[X_train.data for X_train in model.parameters()]
    
    #Create augmented weights+bias tensor for 2nd (last-but-1) hidden layer
    aug = torch.cat([layers[2],layers[3].unsqueeze(1)],1)
    
    # Calculate and add accuracy to the list
    acc = torch.div(torch.sum(torch.eq(model.predict(X_train),y_train)),torch.Tensor([y_train.numel()]))
    accuracy.append(acc.item())
    
    # Compute FP loss
    loss = fp_loss(activations,aug) 
    loss_ = criterion(y_pred,y_train)
    losses.append(loss_.item()) #Add loss to the list
    
    # Print loss and acc during training (every 100 epochs)
    if i % 100 == 0:  
        print("Loss: ",round(loss_.item(),4))
        print("Training Accuracy: ",round(acc.item(),4))
        #print(loss_)
    
    #Clear the previous gradients
    optimizer.zero_grad() 
    #Compute gradients
    loss_.backward()
    #Adjust weights
    optimizer.step()

end = time.time()    
print("Training Time: ",end - start)

# Plot of Losses vs Epochs
f1 = plt.figure()
plt.plot(losses)
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.suptitle("Flip Probability: Loss vs Epochs",fontsize=14,y=0.975)
plt.title("Binary Classification of Wisconsin Breast Cancer Dataset",fontsize=10,y=1)

# Plot of Accuracy vs Epochs
f2 = plt.figure()
plt.plot(accuracy)
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.suptitle("Flip Probability: Training Accuracy vs Epochs",fontsize=14,y=0.975)
plt.title("Binary Classification of Wisconsin Breast Cancer Dataset",fontsize=10,y=1)
plt.ylim(0,1)
plt.show()

# Compute accuracy on test data set
test_acc = torch.div(torch.sum(torch.eq(model.predict(X_test),y_test)),torch.Tensor(np.array([y_test.numel()])))
print("Test Accuracy: ",round(test_acc.item(),4))
