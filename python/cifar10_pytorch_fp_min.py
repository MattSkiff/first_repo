import torch
import torchvision

# parameters
n_epochs = 10
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.1
import random
random_seed = random.randint(1,9999)
torch.manual_seed(random_seed)

# logging frequency (in minibatches)
log_interval = 50

train_loader = torch.utils.data.DataLoader(
        # normalise for 0.5 across the three input channels
  torchvision.datasets.CIFAR10('G:\\My Drive\\data\\tmp\\', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                (0.5,0.5,0.5),(0.5,0.5,0.5))])), 
                                batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.CIFAR10('G:\\My Drive\\data\\tmp\\', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                               (0.5,0.5,0.5), (0.5,0.5,0.5))])),
                               batch_size=batch_size_test, shuffle=True)
                               
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,400)
        x = F.relu(self.fc1(x))
        
        #Returning INPUT to second (last-but-1) hidden layer, not output
        out = x
        
        x = F.relu(self.fc2(x))
        x = self.fc3(x)        
        return F.log_softmax(x),out
    
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
  
  for batch_idx, (data, target) in enumerate(train_loader):  
    for i in range(2): # for each of the loss functions
        n_iter += 1
        if (i == 0):
            network.conv1.weight.requires_grad_ = True
            #network.pool.weight.requires_grad_ = True
            network.conv2.weight.requires_grad_ = True
            network.fc1.weight.requires_grad_ = False
            network.fc2.weight.requires_grad_ = False
            network.fc3.weight.requires_grad_ = False

            
            network.conv1.bias.requires_grad_ = True
            #network.pool.bias.requires_grad_ = True
            network.conv2.bias.requires_grad_ = True
            network.fc1.bias.requires_grad_ = False
            network.fc2.bias.requires_grad_ = False
            network.fc3.bias.requires_grad_ = False

        else:
            network.conv1.weight.requires_grad_ = False
            #network.pool.weight.requires_grad_ = False
            network.conv2.weight.requires_grad_ = False
            network.fc1.weight.requires_grad_ = True
            network.fc2.weight.requires_grad_ = True
            network.fc3.weight.requires_grad_ = True

            
            network.conv1.bias.requires_grad_ = False
           # network.pool.bias.requires_grad_ = False
            network.conv2.bias.requires_grad_ = False
            network.fc1.bias.requires_grad_ = True
            network.fc2.bias.requires_grad_ = True
            network.fc3.bias.requires_grad_ = True

    
        preds,out = network.forward(data) # Forward propagation
        
        if (i == 0):
            optimizer1.zero_grad() # zero out gradients
            layers=[data.data for data in network.parameters()] #Get weights matrices and biases of network as a list
            hi = torch.cat([layers[6],layers[7].unsqueeze(1)],1)
            ones = torch.ones(target.numel()).unsqueeze(1)
            xj = torch.cat([out,ones],1)
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
      hi = torch.cat([layers[6],layers[7].unsqueeze(1)],1) # from fc2
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
for epoch in range(1, n_epochs + 1): 
  train(epoch)
  n_test += 1
  test(n_test)
  
import matplotlib.pyplot as plt
  
# Plot of Train Losses vs Epochs
f1 = plt.figure()
plt.plot(train_losses_fp)
plt.ylabel('Loss')
plt.xlabel('Batch Epochs')
plt.suptitle("Flip Probability: Train FP Loss vs Epochs",fontsize=14,y=0.975)
plt.title("Classification of CIFAR-10",fontsize=10,y=1)
plt.savefig('figs/train_losses_fp.png')

# Plot of Train Losses vs Epochs
f2 = plt.figure()
plt.plot(train_losses_ce)
plt.ylabel('Loss')
plt.xlabel('Batch Epochs')
plt.suptitle("Flip Probability: Train CE Loss vs Epochs",fontsize=14,y=0.975)
plt.title("Classification of CIFAR-10",fontsize=10,y=1)
plt.savefig('figs/train_losses_ce.png')

# Plot of Train Accuracy vs Epochs
train_accuracies_only = [item[0] for item in train_accuracies]
f3 = plt.figure()
plt.plot(train_accuracies_only)
plt.ylabel('Accuracy')
plt.xlabel('Batch Epochs')
plt.suptitle("Flip Probability: Train Accuracy vs Epochs",fontsize=14,y=0.975)
plt.title("Classification of CIFAR-10",fontsize=10,y=1)
plt.savefig('figs/train_accuracy.png')

# Plot of Test Losses vs Epochs
f4 = plt.figure()
plt.plot(test_losses)
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.suptitle("Flip Probability: Test FP Loss vs Epochs",fontsize=14,y=0.975)
plt.title("Classification of CIFAR-10",fontsize=10,y=1)
plt.savefig('figs/test_losses_fp.png')

# Plot of Test Accuracies vs Epochs
f5 = plt.figure()
plt.plot(test_accuracies)
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.suptitle("Flip Probability: Test Accuracy vs Epochs",fontsize=14,y=0.975)
plt.title("Classification of CIFAR-10",fontsize=10,y=1)
plt.savefig('figs/test_accuracy_fp.png')