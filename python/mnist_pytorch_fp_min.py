# original script author: https://nextjournal.com/gkoehler/pytorch-mnist
# adapted and updated to use flip probability

import torch
import torchvision
from matplotlib import pyplot as plt
import time

# parameters
n_epochs = 100
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.0
import random
random_seed = random.randint(1,9999)
torch.manual_seed(random_seed)

# logging frequency (in minibatches)
log_interval = 50

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
        
        #Returning INPUT to second (last-but-1) hidden layer, not output
        out = x
        
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x),out # return both input to fp loss and regular soft-maxed output preds
    
network = Net()

train_counter = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

# the weights (including bias term) from the last-but-1 hidden layer
# the output of the last-but-1 hidden layer, after negative angles are made positive
def fp_loss(hi, xj):
    hi_xj = torch.mm(hi,xj.t()) # the matrix multiplication 
    norm_hi = torch.norm(hi,dim = 1)
    norm_xj = torch.norm(xj,dim = 1)
    fp_loss = torch.sum(torch.pow(torch.abs(torch.div(hi_xj,torch.mm(torch.unsqueeze(norm_hi,1),torch.unsqueeze(norm_xj,1).t()))),2))
    return fp_loss

#List to store losses / accuracy
train_losses_fp = []
train_losses_ce = []
train_accuracies = []
torch_y_numel = torch.Tensor([batch_size_train])

def train(epoch):
  n_iter = 0
  correct = 0
  network.train()
   
  # set optimiser to only calculate gradients for parameters to be updated    
  optimizer1 = optim.SGD([{'params': network.conv1.parameters()},{'params': network.conv2.parameters()}], lr=learning_rate,momentum=momentum)
  optimizer2 = optim.SGD([{'params': network.fc1.parameters()},{'params': network.fc2.parameters()}], lr=learning_rate,momentum=momentum)
  
  # alternating freezing weights as loss is switched between ce/fp
  for batch_idx, (data, target) in enumerate(train_loader):  
    for i in range(2): # for each of the loss functions
        n_iter += 1
        if (i == 0):
            network.conv1.weight.requires_grad_ = True
            network.conv2.weight.requires_grad_ = True
            network.fc1.weight.requires_grad_ = False
            network.fc2.weight.requires_grad_ = False
            
            network.conv1.bias.requires_grad_ = True
            network.conv2.bias.requires_grad_ = True
            network.fc1.bias.requires_grad_ = False
            network.fc2.bias.requires_grad_ = False
        else:
            network.conv1.weight.requires_grad_ = False
            network.conv2.weight.requires_grad_ = False
            network.fc1.weight.requires_grad_ = True
            network.fc2.weight.requires_grad_ = True
            
            network.conv1.bias.requires_grad_ = False
            network.conv2.bias.requires_grad_ = False
            network.fc1.bias.requires_grad_ = True
            network.fc2.bias.requires_grad_ = True
    
        preds,out = network.forward(data) # Forward propagation
        
        if (i == 0):
            optimizer1.zero_grad() # zero out gradients
            layers=[data.data for data in network.parameters()] #Get weights matrices and biases of network as a list
            hi = torch.cat([layers[4],layers[5].unsqueeze(1)],1)
            ones = torch.ones(target.numel()).unsqueeze(1)
            xj = torch.cat([out,ones],1) # manually append bias term to loss 
            loss = fp_loss(hi,xj)
            train_losses_fp.append(loss.item()) #Add loss to the list
            loss.backward() # back propagate loss
            optimizer1.step() # adjust weights matrices
        else:
            optimizer2.zero_grad() 
            loss = F.nll_loss(preds, target)
            train_losses_ce.append(loss.item()) #Add loss to the list
            loss.backward() # back propagate loss
            optimizer2.step() # adjust weights matrices
        
        pred = preds.data.max(1, keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).sum()
        train_accuracy = correct, 64,100. * (correct / 64)
        train_accuracies.append(train_accuracy)

        if batch_idx % log_interval == 0:
            if i == 0:
                string = "FP"
            else:
                string = "CE"
            print(string,'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
            pred = preds.data.max(1, keepdim=True)[1]
            correct = pred.eq(target.data.view_as(pred)).sum()
            train_accuracy = correct, 64,100. * (correct / 64)
            print(string, 'Train set: Accuracy: {}/{} ({:.0f}%)\n'.format(
             correct, 64,100. * correct / 64))

test_losses = []
test_accuracies = []      
        
def test(n_test):
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      layers=[data.data for data in network.parameters()]
      hi = torch.cat([layers[4],layers[5].unsqueeze(1)],1)
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
  print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
    test_loss, correct, len(test_loader.dataset),
    test_accuracy))

n_test = 0
test(n_test)

t0 = time.time() # starting timer

for epoch in range(1, n_epochs + 1): 
  train(epoch)
  n_test += 1
  test(n_test)
  
time.elapsed = time.time() - t0
print('{} seconds'.format(time.elapsed))

# plots section  
  
# Plot of Train Losses vs Epochs
fig1 = plt.figure()
plt.plot(train_losses_fp)
plt.ylabel('Loss')
plt.xlabel('Batch Epochs')
plt.suptitle("Flip Probability: Train FP Loss vs Epochs",fontsize=14,y=0.975)
plt.title("Classification of MNIST",fontsize=10,y=1)
plt.savefig('figs/train_losses_fp.png')

# Plot of Train Losses vs Epochs
fig2 = plt.figure()
plt.plot(train_losses_ce)
plt.ylabel('Loss')
plt.xlabel('Batch Epochs')
plt.suptitle("Flip Probability: Train CE Loss vs Epochs",fontsize=14,y=0.975)
plt.title("Classification of MNIST",fontsize=10,y=1)
plt.savefig('figs/train_losses_ce.png')

# Plot of Train Accuracy vs Epochs
train_accuracies_only = [item[0] for item in train_accuracies] 
f3 = plt.figure()
plt.plot(train_accuracies_only)
plt.ylabel('Accuracy')
plt.xlabel('Batch Epochs')
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