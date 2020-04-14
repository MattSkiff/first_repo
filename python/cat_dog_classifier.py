"""
https://github.com/mukul54/A-Simple-Cat-vs-Dog-Classifier-in-Pytorch/blob/master/catVsDog.py

Getting a binary classifier working on a reasonably large data set
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import matplotlib.pyplot as plt
import torchvision
import os
import csv
import glob
import cv2
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import imageio
from skimage import transform,io
from random import shuffle
import glob
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
class CatDogs(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.catdogs = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.catdogs)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
    
        img_name = os.path.join(self.root_dir,self.catdogs.iloc[idx, 0])
        image = io.imread(img_name)
        
        catdogs_labels = self.catdogs.iloc[idx, 1:]
        catdogs_labels = np.array([catdogs_labels])
        catdogs_labels = catdogs_labels.astype('float').reshape(-1, 2)
        sample = {'image': image, 'label': catdogs_labels}
        
        sample = {'image': image,'label':catdogs_labels}
        
        if self.transform:
            sample = self.transform(sample)

        return sample
    
catdog_dataset = CatDogs(csv_file='C:\\Users\\user\\Desktop\\PetImages\\train\\flist_label.csv',
                                root_dir='C:\\Users\\user\\Desktop\\PetImages\\train\\')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 50, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(50, 100, 7)
        self.pool2 = nn.MaxPool2d(2,2)
        
        self.fc1 = nn.Linear(100 * 12 * 12, 120)
        self.fc2 = nn.Linear(120, 100)
        self.fc3 = nn.Linear(100, 2)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 100 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

net = Net()
print(net)

shuffle_data = True  # shuffle the addresses before saving
cat_dog_train_path = 'C:\\Users\\user\\Desktop\\PetImages\\train\\*.jpg'
# read addresses and labels from the 'train' folder
addrs = glob.glob(cat_dog_train_path)
labels = [ [1,0] if '1_' in addr else [0,1] for addr in addrs]  # 1 = Cat, 0 = Dog
    # to shuffle data
    
if shuffle_data:
    c = list(zip(addrs, labels))
    shuffle(c)
    addrs, labels = zip(*c)
    print(labels[0:10])
    
# Divide the hata into 60% train, 20% validation, and 20% test
train_addrs = addrs[0:int(0.6*len(addrs))]
train_labels = labels[0:int(0.6*len(labels))]
#train_addrs.size

val_addrs = addrs[int(0.6*len(addrs)):int(0.8*len(addrs))]
val_labels = labels[int(0.6*len(addrs)):int(0.8*len(addrs))]

test_addrs = addrs[int(0.8*len(addrs)):]
test_labels = labels[int(0.8*len(labels)):]

# loop over train addresses
train_data = []
for i in range(len(train_addrs[:1000])):
    # print how many images are saved every 10 images
    if i % 1000 == 0 and i > 1:
        print ('Train data: {}/{}'.format(i, len(train_addrs)))
    # read an image and resize to (64, 64)
    # cv2 load images as BGR, convert it to RGB
    addr = train_addrs[i]
    img = imageio.imread(addr)
   # img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
    img = transform.resize(img, (64,64), mode='symmetric', preserve_range=True)
    img = img.astype('float32')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    train_data.append([np.array(img), np.array(train_labels[i])])
    shuffle(train_data)
    np.save('train_data.npy', train_data)
   
    # loop over test addresses
    # creating test data
test_data = []
for i in range(len(test_addrs[:10])):
    # print how many images are saved every 10 images
    if i % 9 == 0 and i > 1:
        print ('Test data: {}/{}'.format(i, len(test_addrs)))
    # read an image and resize to (64, 64)
    # cv2 load images as BGR, convert it to RGB
    addr = test_addrs[i]
    img = imageio.imread(addr)
    img = transform.resize(img, (64,64), mode='symmetric', preserve_range=True)
    #img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
    img = img.astype('float32')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    test_data.append([np.array(img), np.array(labels[i])])
shuffle(test_data)
np.save('test_data.npy', test_data)

# loop over val addresses
val_data = []
for i in range(len(val_addrs[:10])):
    # print how many images are saved every 1000 images
    if i % 9 == 0 and i > 1:
        print ('Val data: {}/{}'.format(i, len(val_addrs)))
    # read an image and resize to (64, 64)
    # cv2 load images as BGR, convert it to RGB
    addr = val_addrs[i]
    img = imageio.imread(addr)
    #img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC) #
    img = transform.resize(img, (64,64), mode='symmetric', preserve_range=True)
    img = img.astype('float32')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    val_data.append([np.array(img), np.array(labels[i])])
shuffle(val_data)
np.save('val_data.npy', val_data)
#print(val_data[1])

X = np.array([i[0] for i in train_data]).reshape(-1,64,64,3)
X = Variable(torch.Tensor(X))
X = X.reshape(-1,64,64,3)
X = X.permute(0,3,1,2)
print(X.shape)
#Y = Variable(torch.Tensor(Y))

Y = np.array([i[1] for i in train_data])
target = Variable(torch.Tensor(Y))
target = target.type(torch.LongTensor)

print(target.shape)
#print(target)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.0001, momentum = 0.9)
for epoch in range(50):
    running_loss  = 0.0
    optimizer.zero_grad() #zero the parameter gradients
    output = net(X)
    
    loss = criterion(output, torch.max(target, 1)[1])
    
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    print(epoch, ':', running_loss)

test = np.array([i[0] for i in test_data]).reshape(-1,64,64,3)
test = Variable(torch.Tensor(test))
test = test.reshape(-1,64,64,3)
test = test.permute(0,3,1,2)
print(test.shape)
#Y = Variable(torch.Tensor(Y))

tlabels = np.array([i[1] for i in test_data])
tlabels = Variable(torch.Tensor(tlabels))
tlabels = tlabels.type(torch.float)

print(tlabels.shape)
print(tlabels)


correct = 0
total = 0
with torch.no_grad():
    for data in zip(X,target):
        images, labels = data
        images = images.reshape(1,3,64,64)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        #total += labels.size(0)
        if((predicted == 0 and labels[0] == 1) or (predicted == 1 and labels[1]==1) ):
            correct+=1
        #correct += (predicted == labels).sum().item()
        #print(outputs,labels)
total = X.shape[0]
print('Train accuracy of the network on the' + str(total) +  'train images: %f %%' % (
    100 * (correct*1.0) / total) )
print(correct, total)

correct = 0
total = 0
with torch.no_grad():
    for data in zip(test,tlabels):
        images, labels = data
        images = images.reshape(1,3,64,64)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        #total += labels.size(0)
        if((predicted == 0 and labels[0] == 1) or (predicted == 1 and labels[1]==1) ):
            correct += 1
            
total = test.shape[0]
print('Test accuracy of the network on the ' + str(total) +  ' test images: %f %%' % (
    100 * (correct*1.0) / total) )
print(correct, total)
