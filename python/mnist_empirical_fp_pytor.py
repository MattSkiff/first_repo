"""
Attempt to implement empirical FP by randomly projecting 50 times and using calculation of labels flips
Back to issue that calculation of flips requires torch.eq operation, which is not differentiable
"""


# train_accuracies
import torch
import torchvision
import numpy as np

n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('G:\\My Drive\\data\\tmp\\', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('G:\\My Drive\\data\\tmp\\', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

import matplotlib.pyplot as plt

fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])
fig

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        
        out = x # sending input to last layer out, for use in FP for error 
        
        # list random projection matrix
        Rm_array = []
        for i in range(20):
            Rm_array.append(torch.from_numpy(np.random.rand(k,layers[6].shape[1]).astype('float32'))) #random projection matrix 
        
        pred_array = []
        for i in range(20):
            p_class = torch.mm(classifier,random_matrices[i].t())
            p_data = torch.mm(data,random_matrices[i].t())
            pred = torch.mm(p_data,p_class.t())
            pred = pred.add(bias)
            #print(F.log_softmax(pred)) # looks about right
            pred_array[i] = pred
            
        #x = self.fc2(x)
        
        return pred_array[i]
    
"""
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        
        out = x # sending input to last layer out, for use in FP for error 
            
        x = self.fc2(x)
        
        return F.log_softmax(x),out
"""
    
network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)


# projection dimension
k = 5

layers=[data.data for data in network.parameters()]

"""
# list random projection matrix
Rm_array = []
for i in range(20):
    Rm_array.append(torch.from_numpy(np.random.rand(k,layers[6].shape[1]).astype('float32'))) #random projection matrix 
"""

"""
# works, but no improvements in accuracy 
# tested with regular setup - very low accuracy - including elements of forward prop in loss function seems to break training
def fp_loss(classifier,data,random_matrices,bias,target,output):
#    loss = 0
#    for i in range(20):
#        p_class = torch.mm(classifier,random_matrices[i].t())
#        p_data = torch.mm(data,random_matrices[i].t())
#        pred = torch.mm(p_data,p_class.t())
#        pred = pred.add(bias)
#        #print(F.log_softmax(pred)) # looks about right
#        loss += F.nll_loss(F.log_softmax(pred),target)
    print(classifier.shape)
    print(data.shape)
    return F.nll_loss(F.log_softmax(torch.mm(data,classifier.t()).add(bias)),target)
"""
# doesn't work, torch.eq not differentiable

def fp_loss(classifier,data,random_matrices,bias,target,output):
    #correct = torch.sum(torch.eq(torch.argmax(output,dim = 1),target)) # returns indice across correct dimension (outputs 0-9)
    loss = 0
    for i in range(20):
        p_class = torch.mm(classifier,random_matrices[i].t()) 
        p_data = torch.mm(data,random_matrices[i].t())
        pred = torch.mm(p_data,p_class.t())
        pred = pred.add(bias)
        #print(F.log_softmax(pred)) # looks about right
        loss += torch.sum(torch.eq(torch.argmax(pred,dim = 1),target))
    
    return loss
 """

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    layers=[data.data for data in network.parameters()] # updating storage of weights matrices and bias to use in FP loss
    optimizer.zero_grad
    output,input_fc2 = network(data)
    #loss = F.nll_loss(output, target)
    loss = fp_loss(classifier = layers[6],data = input_fc2,random_matrices = Rm_array,bias = layers[7],target = target,output = output)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      
def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad(): 
    for data, target in test_loader:
      output = network(data)[0]
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
  
test()
for epoch in range(1, n_epochs + 1):
  train(epoch)
  test()
  
fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
fig

with torch.no_grad():
  output = network(example_data)[0]
  
fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Prediction: {}".format(
    output.data.max(1, keepdim=True)[1][i].item()))
  plt.xticks([])
  plt.yticks([])
fig