# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 00:07:25 2019
https://medium.com/@prudhvirajnitjsr/simple-classifier-using-pytorch-37fba175c25c
@author: pmaddala

Adapted to use FP loss
"""

import sklearn.datasets
import torch
import numpy as np

np.random.seed(0)
X, y = sklearn.datasets.make_moons(200,noise=0.2)
#diabetes = datasets.load_diabetes()

import matplotlib.pyplot as plt

plt.scatter(X[:,0],X[:,1],s=40,c=y,cmap=plt.cm.binary)

X = torch.from_numpy(X).type(torch.FloatTensor)
y = torch.from_numpy(y).type(torch.LongTensor)

import torch.nn as nn
import torch.nn.functional as F

#our class must extend nn.Module
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        #Our network consists of 3 layers. 1 input, 1 hidden and 1 output layer
        #This applies Linear transformation to input data. 
        self.fc1 = nn.Linear(2,3)
        
        #This applies linear transformation to produce output data
        self.fc2 = nn.Linear(3,2)
        
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
        return torch.tensor(ans)
       #Initialize the model        
model = Net()

#Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#custom loss function
def fp_loss(activations, last_but_one_classifier):
    hi = last_but_one_classifier
    xj = activations
    hi_xj = torch.mm(hi,xj.t())
    norm_hi = torch.norm(hi,dim = 1)
    norm_xj = torch.norm(xj,dim = 1)
    fp_loss = torch.sum(torch.pow(torch.abs(torch.div(hi_xj,torch.mm(torch.unsqueeze(norm_hi,1),torch.unsqueeze(norm_xj,1).t()))),2))
    return fp_loss

from sklearn.metrics import accuracy_score

# original script loss
criterion = nn.CrossEntropyLoss()

#Number of epochs
epochs = 500
#List to store losses / accuracy
losses = []
accuracy = []
for i in range(epochs):
    #Predict the output for Given input
    y_pred = model.forward(X)[0]
    
    #Get Last-but-1 layer output on forward pass
    activations = model.forward(X)[1]
    
    #Get weights and bias as list
    layers=[X.data for X in model.parameters()]
    
    #Create augmented weights+bias tensor
    aug = torch.cat([layers[0],layers[1].unsqueeze(1)],1)
    
    #Compute Cross entropy loss
    #loss = criterion(y_pred,y)
    # Compute FP loss
    loss = fp_loss(activations,aug)
    if i % 500 == 0:  
        print(loss)
        #print(loss_)
    #Add loss to the list
    losses.append(loss.item())
    
    # Calculate and add accuracy to the list
    acc = accuracy_score(model.predict(X),y)
    accuracy.append(acc.item())
    
    #Clear the previous gradients
    optimizer.zero_grad()
    #Compute gradients
    loss.backward()
    #Adjust weights
    optimizer.step()
    
print(accuracy_score(model.predict(X),y))
    
def predict(x):
    x = torch.from_numpy(x).type(torch.FloatTensor)
    ans = model.predict(x)
    return ans.numpy()

# Helper function to plot a decision boundary.
# If you don't fully understand this function don't worry, it just generates the contour plot below.
def plot_decision_boundary(pred_func,X,y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx,yy=np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.binary)
    
#f3 = plt.figure() 
plot_decision_boundary(lambda x : predict(x) ,X.numpy(), y.numpy())

# Plot of Losses vs Epochs
f1 = plt.figure()
plt.plot(losses)
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.suptitle("Flip Probability: Loss vs Epochs",fontsize=14,y=0.975)
plt.title("Binary Classification of Synthetic 'Moon' Dataset",fontsize=10,y=1)

# Plot of Accuracy vs Epochs
f2 = plt.figure()
plt.plot(accuracy)
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.suptitle("Flip Probability: Accuracy vs Epochs",fontsize=14,y=0.975)
plt.title("Binary Classification of Synthetic 'Moon' Dataset",fontsize=10,y=1)
plt.ylim(0,1)
plt.show()
