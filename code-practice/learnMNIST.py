#%%
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time 
#from torchvision import transforms
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn, optim

#%%
PATH_TRAINSET = './trainMNIST'
PATH_TESTSET = './testMNIST'

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])

trainset = torchvision.datasets.MNIST(PATH_TRAINSET, train=False, transform = transform, download=True)

testset = torchvision.datasets.MNIST(PATH_TESTSET, train=True, transform=transform, download=True)

# %%
train_loader = DataLoader(trainset, batch_size=64, shuffle= True)
test_loader = DataLoader(testset, batch_size=64, shuffle=True)

#%%
train_iter = iter(train_loader)
image, label = train_iter.next()

# %%
#plt.imshow(image[0].numpy().squeeze(), cmap='gray_r')

# %%
input_size = 784
hidden_size=[128, 64]
output_size = 10

model = nn.Sequential(nn.Linear(input_size, hidden_size[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_size[0], hidden_size[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_size[1], output_size),
                      nn.LogSoftmax(dim=1))
print(model)

# %%
criterion = nn.NLLLoss()
images, labels = next(iter(train_loader))
images = images.view(images.shape[0], -1)

# %%
log_ps = model(images)
loss = criterion(log_ps,labels)

# %%
print('Before backward pass: \n', model[0].weight.grad)
loss.backward()
print('After backward pass: \n', model[0].weight.grad)


# %%
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
time0 = time()
epochs = 15
for e in range(epochs):
    running_loss = 0
    for images, labels in train_loader: 
        images = images.view(images.shape[0], -1)
        optimizer.zero_grad()
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    else:
        print("Epoch {} - Trainning loss: {} - This Loss:{}".format(e, running_loss/len(train_loader), loss.item()))
print("\nTrainning Time (in minutes) =", (time() - time0)/60)


# %%
