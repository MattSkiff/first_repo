"""
Original Script: https://nextjournal.com/gkoehler/pytorch-mnist
Author: Gregor Koehler

Adapted to;

* use multiple loss functions via alternating training scheme (freeze/unfreeze layers)
* alternates between loss over every batch of data, sequentially - per epoch
* runs single optimiser

"""

import torch
import torchvision
import torchvision.transforms as transforms

# using cuda
use_cuda = torch.cuda.is_available()

from torch.utils.tensorboard import SummaryWriter

# tensorboard
# Writer will output to ./runs/ directory by default
# must be in python scripts folder (parent folder of runs)
# tensorboard --logdir=runs -> Run on command line to start tensorboard -> http://localhost:6006/

# tensor board?
tb = False
# print grads?
use_cuda = False

# diagnostic flags
diag = False
print_grads = False
check_weights = False
all_zero = False

# parameters
n_epochs = 10
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.0
import random
random_seed = random.randint(1,9999)
torch.manual_seed(random_seed)


# logging frequency (in minibatches)
log_interval = 50

torch.backends.cudnn.enabled = True

if tb:
    writer = SummaryWriter()

device = torch.device('cuda' if use_cuda else 'cpu')

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.CIFAR10('G:\\My Drive\\data\\tmp\\', train=True, download=True,
                             transform=transform),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.CIFAR10('G:\\My Drive\\data\\tmp\\', train=False, download=True,
                             transform=transform),
  batch_size=batch_size_test, shuffle=True)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
                               
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        
        #Returning INPUT to second (last-but-1 i.e. last-but-output layer) hidden layer, not output
        out = x
        
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x,out
    
network = Net()
#optimizer = optim.SGD(network.parameters(), lr=learning_rate,momentum=momentum)

train_counter = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

# the weights (including bias term) from the last-but-1 hidden layer
# the output of the last-but-1 hidden layer, after going through relu activation
def fp_loss(hi, xj):
    hi_xj = torch.mm(hi,xj.t()) # the matrix multiplication 
    norm_hi = torch.norm(hi,dim = 1)
    norm_xj = torch.norm(xj,dim = 1)
    #fp_loss = torch.sum(torch.pow(F.relu(torch.div(hi_xj,torch.mm(torch.unsqueeze(norm_hi,1),torch.unsqueeze(norm_xj,1).t()))),2)) # Relu version - works
    fp_loss = torch.sum(torch.pow(torch.abs(torch.div(hi_xj,torch.mm(torch.unsqueeze(norm_hi,1),torch.unsqueeze(norm_xj,1).t()))),2))
    return fp_loss

# Check trainable variables

# Model Summary
from torchsummary import summary
# TODO: fix broken summary 
#summary(network,torch.Size([64,1,28,28]),device = str(device)) 
print(network)

#List to store losses / accuracy
train_losses_fp = []
train_losses_ce = []
train_accuracies = []
torch_y_numel = torch.Tensor([batch_size_train])

import copy

criterion = nn.CrossEntropyLoss()

def train(epoch,i):
  n_iter = 0
  correct = 0
  network.train()
  # for each batch in the training data
  optimizer = optim.SGD(filter(lambda p: p.requires_grad_, network.parameters()), lr=learning_rate,momentum=momentum)
# for each of the loss functions
  for batch_idx, (data, target) in enumerate(train_loader):  
        n_iter += 1
        if (i == 0):
            network.conv1.weight.requires_grad_ = True
            network.conv2.weight.requires_grad_ = True
            network.fc1.weight.requires_grad_ = False
            network.fc2.weight.requires_grad_ = False
            network.fc3.weight.requires_grad_ = False
            
            network.conv1.bias.requires_grad_ = True
            network.conv2.bias.requires_grad_ = True
            network.fc1.bias.requires_grad_ = False
            network.fc2.bias.requires_grad_ = False
            network.fc3.bias.requires_grad_ = False
        else:
            network.conv1.weight.requires_grad_ = False
            network.conv2.weight.requires_grad_ = False
            network.fc1.weight.requires_grad_ = True
            network.fc2.weight.requires_grad_ = True
            network.fc3.weight.requires_grad_ = True
            
            network.conv1.bias.requires_grad_ = False
            network.conv2.bias.requires_grad_ = False
            network.fc1.bias.requires_grad_ = True
            network.fc2.bias.requires_grad_ = True
            network.fc3.bias.requires_grad_ = True
              
        # clear gradients
        optimizer.zero_grad() 
        
        # works, not very well
        optimizer = optim.SGD(filter(lambda p: p.requires_grad_, network.parameters()), lr=learning_rate,momentum=momentum)
    
        # Forward propagation
        preds,out = network.forward(data)
        
        if (i == 0):
            #Get weights matrices and biases of network as a list
            layers=[data.data for data in network.parameters()]
            hi = torch.cat([layers[6],layers[7].unsqueeze(1)],1)
            ones = torch.ones(target.numel()).unsqueeze(1)
            xj = torch.cat([out,ones],1)
            loss = fp_loss(hi,xj)
            train_losses_fp.append(loss.item()) #Add loss to the list
            if tb:
                writer.add_scalar('LossFP/train',loss,n_iter)
        else:
            loss = criterion(preds, target)
            train_losses_ce.append(loss.item()) #Add loss to the list
            if tb:
                writer.add_scalar('LossCE/train',loss,n_iter)
        
        if check_weights:
            before = copy.deepcopy(list(network.parameters()))
        
        loss.backward() # back propagate loss
        optimizer.step() # adjust weights matrices
        
        if diag == True and n_iter > 2:
            print("\n")
            for name,param in network.named_parameters():
                    print("Name: ",name)
                    print("Requires grad: ",param.requires_grad_)
                    print("Sum of Weights: ",param.data.sum().item())
                    print("Sum of Gradients: ",param.grad.sum().item())
            print("\n")
        
        pred = preds.data.max(1, keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).sum()
        train_accuracy = 100. * correct / len(data)
        train_accuracies.append(train_accuracy.item())
    
        if tb:
            writer.add_scalar('Accuracy/train',train_accuracy)
        
        if check_weights:
            after = list(network.parameters())
    
        if check_weights:
            if batch_idx % 2 == 0:
                print("Weights Equality? FP")
                print("\n")
                for i in range(len(before)):
                    print(torch.equal(before[i].data, after[i].data))
            else: 
                print("Weights Equality? CE")
                print("\n")
                for i in range(len(before)):
                    print(torch.equal(before[i].data, after[i].data))
            print("\n")
            print("\n")
            

        if batch_idx % log_interval == 0:
            if i == 0:
                string = "FP"
            else:
                string = "CE"
            print(string,' Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))  
            print(string, 'Train set: Accuracy: {}/{} ({:.0f}%)\n'.format(
             correct, 64,100. * correct / 64))
            #train_accuracies.append(train_accuracy)

        if all_zero and batch_idx > 2:
            print('\nAll Zero Grads?')
            print('\nConv1 Grads')
            print(network.conv1.weight.grad.sum().item() == 0)
            print('\nConv2 Grads')
            print(network.conv2.weight.grad.sum().item() == 0)
            print('\nFC1 Grads') 
            print(network.fc1.weight.grad.sum().item() == 0) # returns none
            print('\nFC2 Grads')
            print(network.fc2.weight.grad.sum().item() == 0) # returns none
        
        if print_grads:
            if batch_idx % log_interval == 0:
              print('\nConv1 Grads')
              print(network.conv1.weight.grad)
              print('\nConv2 Grads')
              print(network.conv2.weight.grad)
              print('\nFC1 Grads') 
              print(network.fc1.weight.grad) # returns none
              print('\nFC2 Grads')
              print(network.fc2.weight.grad) # returns none


test_losses = []
test_accuracies = []      
        
def test(n_test):
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      layers=[data.data for data in network.parameters()]
      hi = torch.cat([layers[6],layers[7].unsqueeze(1)],1)
      ones = torch.ones(target.numel()).unsqueeze(1)
      xj = torch.cat([network.forward(data)[1],ones],1)
      output = network(data)[0]
      test_loss += fp_loss(hi,xj)
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  test_accuracy = 100. * correct / len(test_loader.dataset)
  test_accuracies.append(test_accuracy)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    test_accuracy))
  if tb:
      writer.add_scalar('Accuracy/train',test_accuracy,n_test)
      writer.add_scalar('LossFP/test',test_loss,n_test)

n_test = 0
test(n_test)
for i in range(2):
    for epoch in range(1, n_epochs + 1): 
      train(epoch,i)
      n_test += 1
      test(n_test)
  
import matplotlib.pyplot as plt
  
# Plot of Train Losses vs Epochs
f1 = plt.figure()
plt.plot(train_losses_fp)
plt.ylabel('Loss')
plt.xlabel('Batches')
plt.suptitle("Flip Probability: Train FP Loss vs Epochs",fontsize=14,y=0.975)
plt.title("Classification of MNIST",fontsize=10,y=1)
plt.savefig('figs/train_losses_fp.png')

# Plot of Train Losses vs Epochs
f2 = plt.figure()
plt.plot(train_losses_ce)
plt.ylabel('Loss')
plt.xlabel('Batches')
plt.suptitle("Flip Probability: Train CE Loss vs Epochs",fontsize=14,y=0.975)
plt.title("Classification of MNIST",fontsize=10,y=1)
plt.savefig('figs/train_losses_ce.png')

# Plot of Train Accuracy vs Epochs
f3 = plt.figure()
plt.plot(train_accuracies)
plt.ylabel('Accuracy')
plt.xlabel('Batches')
plt.suptitle("Flip Probability: Train Accuracy vs Epochs",fontsize=14,y=0.975)
plt.title("Classification of MNIST",fontsize=10,y=1)
plt.savefig('figs/train_accuracy.png')

# Plot of Test Losses vs Epochs
f4 = plt.figure()
plt.plot(test_losses)
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.suptitle("Flip Probability: Test FP Loss vs Epochs",fontsize=14,y=0.975)
plt.title("Classification of MNIST",fontsize=10,y=1)
plt.savefig('figs/test_losses_fp.png')

# Plot of Test Accuracies vs Epochs
f5 = plt.figure()
plt.plot(test_accuracies)
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.suptitle("Flip Probability: Test Accuracy vs Epochs",fontsize=14,y=0.975)
plt.title("Classification of MNIST",fontsize=10,y=1)
plt.savefig('figs/test_accuracy_fp.png')

if tb:
    writer.close()

# Testing the weights are being updated
# https://stackoverflow.com/questions/51104648/pytorch-gradients-exist-but-weights-not-updating
#import copy
#before = copy.deepcopy(list(network.parameters())) # .copy() makes a deep copy of the weights etc
#train(epoch)
#after = list(network.parameters())
#for i in range(len(before)):
#    print(torch.equal(before[i].data, after[i].data))
#    
## check parameter names and gradients
#for name,param in network.named_parameters():
#    print(name,param.data)