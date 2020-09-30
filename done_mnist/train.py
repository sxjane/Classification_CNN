#%%
import torchvision
from torchvision.datasets import MNIST 
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import MnistCNN

# %%
#download MNIST DATA
train_set = MNIST('./', train=True, transform=transforms.Compose([transforms.ToTensor()]),download=True)
test_set = MNIST('./', train=False, transform=transforms.Compose([transforms.ToTensor()]),download=True)
# %%
#display a sample
plt.imshow(train_set[0][0].squeeze(), cmap='gray')

# %%
#load data with batch size 100
train_loader = DataLoader(train_set, batch_size=100)
test_loader = DataLoader(test_set, batch_size=100)

# %%
#setting of MNISTCNN
model = MnistCNN.MnistCNN()
error = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

iterations_list = []
loss_list = []
count = 0
epoch = 20  ####

#%%
model = model.cuda() ####
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) ####

for i in range(epoch):
    for images, labels in train_loader:
        images = images.cuda()   ####
        labels = labels.cuda()   ####
        outputs = model(images)
        loss = error(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count +=1
        if not(count%50):
            loss_list.append(loss.data.item())
            iterations_list.append(count)
    corrects = 0
    total = 0
    for images, labels in test_loader:
        images = images.cuda()   ####
        labels = labels.cuda()   ####
        outputs = model(images)
        loss = error(outputs, labels).item()  ####
        predictions = torch.max(outputs, 1)[1]
        corrects += (labels == predictions).sum().item()
        total += len(labels)
    print('After the epoch {}, accuracy is {}, loss is {}'.format(
        i+1, 
        corrects/total,
        loss)) ####
        
# %%
plt.plot(iterations_list, loss_list)
plt.xlabel('Num of Iterations')
plt.ylabel('Losses')
plt.show()

# %%
