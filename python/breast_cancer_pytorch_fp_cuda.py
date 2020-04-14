# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 00:07:25 2019
https://medium.com/@prudhvirajnitjsr/simple-classifier-using-pytorch-37fba175c25c
@author: pmaddala

Adapted to use FP loss
Adapted to use the breast cancer dataset from sklearn
"""

import sklearn.datasets
import torch
import numpy as np
from torch.autograd import Variable
import time

np.random.seed(0)
#X, y = sklearn.datasets.make_moons(200,noise=0.2)
X,y = sklearn.datasets.load_breast_cancer(return_X_y = True)

import matplotlib.pyplot as plt

#plt.scatter(X[:,0],X[:,1],s=40,c=y,cmap=plt.cm.binary)

# TODO: code for visualising data here

torch.set_default_tensor_type(torch.FloatTensor)
X = torch.from_numpy(X).type(torch.FloatTensor).cuda()
y = torch.from_numpy(y).type(torch.FloatTensor).cuda()

import torch.nn as nn
import torch.nn.functional as F

#our class must extend nn.Module
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        #Our network consists of 3 layers. 1 input, 1 hidden and 1 output layer
        #This applies Linear transformation to input data. 
        self.fc1 = nn.Linear(in_features = 30,out_features = 15)
        
        #This applies linear transformation to produce output data
        self.fc2 = nn.Linear(15,2)
        
    #This must be implemented
    def forward(self,x):
        #Output of the first layer
        x = self.fc1(x)
        #Activation function is Relu. Feel free to experiment with this
        x = F.relu(x)
        
        out = x
        
        #This produces output
        x = self.fc2(x)
        return x, out
        
    #This function takes an input and predicts the class, (0 or 1)        
    def predict(self,x):
        #Apply softmax to output
        pred = F.softmax(self.forward(x)[1])
        ans = []
        for t in pred:
            if t[0]>t[1]:
                ans.append(0)
            else:
                ans.append(1)
        return torch.tensor(ans).cuda()
       #Initialize the model        
model = Net()
model.cuda()

#Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#custom loss function
def fp_loss(activations, last_but_one_classifier):
    hi = last_but_one_classifier
    xj = activations
    hi_xj = torch.mm(hi.t(),xj.t())
    norm_hi = torch.norm(hi,dim = 1)
    norm_xj = torch.norm(xj,dim = 0)
    fp_loss = torch.sum(torch.pow((torch.div(hi_xj,torch.dot(norm_hi,norm_xj))),2))
    return fp_loss

from sklearn.metrics import accuracy_score

# original script loss
#criterion = nn.CrossEntropyLoss()

#Number of epochs
epochs = 500
#List to store losses / accuracy
losses = []
accuracy = []

start = time.time()
y_len = torch.Tensor([[y.numel()]])

for i in range(epochs):
    #Predict the output for Given input
    y_pred = model.forward(X)[0]
    
    #Get Last-but-1 layer output on forward pass
    activations = model.forward(X)[1]
    
    #Get weights and bias as list
    layers=[X.data for X in model.parameters()]
    
    #Create augmented weights+bias tensor
    aug = torch.cat([layers[0],layers[1].unsqueeze(1)],1)
    
    # Calculate and add accuracy to the list
    acc = torch.div(torch.sum(torch.eq(model.predict(X),y)),y_len)#torch.Tensor(np.array([y.numel()])))
    accuracy.append(acc.item())
    
    #Compute Cross entropy loss
    #loss = criterion(y_pred,y)
    
    # Compute FP loss
    loss = fp_loss(activations,aug) 
    losses.append(loss.item()) #Add loss to the list
    
    # Print loss and acc during training
    if i % 20 == 0:  
        print(loss)
        print(acc)
        #print(loss_)
    
    #Clear the previous gradients
    optimizer.zero_grad() 
    #Compute gradients
    loss.backward()
    #Adjust weights
    optimizer.step()

end = time.time()    
print(end - start)
#print(accuracy_score(model.predict(X),y))
    
def predict(x):
    x = torch.from_numpy(x).type(torch.FloatTensor)
    ans = model.predict(x)
    return ans.numpy()

## Plot of Losses vs Epochs
#f1 = plt.figure()
#plt.plot(losses)
#plt.ylabel('Loss')
#plt.xlabel('Epochs')
#plt.suptitle("Flip Probability: Loss vs Epochs",fontsize=14,y=0.975)
#plt.title("Binary Classification of Synthetic 'Moon' Dataset",fontsize=10,y=1)
#
## Plot of Accuracy vs Epochs
#f2 = plt.figure()
#plt.plot(accuracy)
#plt.ylabel('Accuracy')
#plt.xlabel('Epochs')
#plt.suptitle("Flip Probability: Accuracy vs Epochs",fontsize=14,y=0.975)
#plt.title("Binary Classification of Synthetic 'Moon' Dataset",fontsize=10,y=1)
#plt.ylim(0,1)
#plt.show()
