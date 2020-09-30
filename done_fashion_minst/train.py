#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision 
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix

#%%
train_csv = pd.read_csv('./fashion-mnist_train.csv')
test_csv = pd.read_csv('./fashion-mnist_test.csv')

# %%
class FashionDataset(Dataset):
    def __init__(self, data, transform = None):
        self.fashion_MNIST = list(data.to_numpy())
        self.transform = transform

        label=[]
        image=[]

        for i in self.fashion_MNIST:
            label.append(i[0])
            image.append(i[1:])
        self.labels = np.asarray(label)
        self.images = np.asarray(image).reshape(-1,28,28,1).astype('float32')

    def __getitem__(self, index):
        label = self.labels[index]
        image = self.images[index]

        if self.transform is not None:
            image = self.transform(image)
        
        return image, label

    def __len__(self):
        return len(self.images)


# %%
train_set = FashionDataset(train_csv, transform=transforms.Compose([transforms.ToTensor()]))
test_set = FashionDataset(test_csv, transform=transforms.Compose([transforms.ToTensor()]))

# %%
batch=1000
train_loader = DataLoader(train_set, batch_size=batch)
test_loader = DataLoader(test_set, batch_size=batch)

# %%
def output_label(label):
    output_mapping = {
        0:'T-shirt/Top',
        1:'Trouser',
        2:'Pullover',
        3:'Dress',
        4:'Coat',
        5:'Sandal',
        6:'Shirt',
        7:'Sneaker',
        8:'Bag',
        9:'Ankle Boot'
    }
    
    input = (label.item() if type(label) == torch.Tensor else label)
    return output_mapping[input]

# %%
# image, label = next(iter(train_set))
# plt.imshow(image.squeeze(),cmap='gray')
# demo_loader = torch.utils.data.DataLoader(train_set, batch_size=10)
# batch = next(iter(demo_loader))
# images, labels = batch
# print(type(images), type(labels))
# print(images.shape, labels.shape)

# grid = torchvision.utils.make_grid(images, nrow=10)
# plt.figure(figsize=(15,20))
# plt.imshow(np.transpose(grid, (1,2,0)))
# print("labels: ", end=" ")
# for i, label in enumerate(labels):
#     print(output_label(label), end=", ")

# %%
class FashionCnn(nn.Module):
    
    def __init__(self):
        super(FashionCnn, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
    
# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = FashionCnn()
model.to(device)
error = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print(model)

# %%
num_epochs = 10
count = 0
loss_list = []
iteration_list = []
accuracy_list = []
predictions_list = []
labels_list = []

for epoch in range(num_epochs):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        #Forward pass
        outputs = model(images)
        loss = error(outputs, labels)
        
        #Initializing a gradient as 0 so there is no mixing of gradient among the batche
        optimizer.zero_grad()

        #Propagating the error backward
        loss.backward()

        #Optimizing the parameters
        optimizer.step()
        count +=1

        #Testing the model
        if not(count % 5):
            total = 0
            correct = 0

            for images, labels in test_loader:
                test, labels = images.to(device), labels.to(device)
                labels_list.append(labels)

                outputs = model(test)
                predictions = torch.max(outputs, 1)[1].to(device)
                predictions_list.append(predictions)
                correct += (predictions == labels).sum()

                total += len(labels)

            accuracy = correct * 100 / total 
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
                    
            if not(count % 50):
                print("Iteration:{}, Loss:{}, Accuracy:{}%".format(count, loss.data, accuracy))

# %%
plt.figure(figsize=(5,5))
plt.plot(iteration_list, loss_list)
plt.xlabel('No. of iteration')
plt.ylabel('Loss')
plt.title('Iteration vs Loss')
plt.show()

#%%
plt.figure(figsize=(5,5))
plt.plot(iteration_list, accuracy_list)
plt.xlabel('No. off iteration')
plt.ylabel('Accuracy')
plt.title('Iteration vs Accuracy')
plt.show()

# %%
import sklearn.metrics as metrics
class_correct = [0. for _ in range(10)]
total_labels = [0. for _ in range(10)]

sk_predictions=[]
sk_labels=[]

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        sk_labels.extend(list(labels[i].item() for i in range(len(labels))))
        outputs = model(images)
        predictions = torch.max(outputs, 1)[1]
        sk_predictions.extend([predictions[i].item() for i in range(len(predictions))])
        corrects = (predictions == labels).squeeze()
        for i in range(1000):
            label = labels[i]
            class_correct[label] += corrects[i].item()
            total_labels[label] += 1
    for i in range(10):
        print('Accuracy of {}: {:.2f}%'.format(output_label(i), class_correct[i]*100/total_labels[i]))

    print(metrics.classification_report(sk_labels, sk_predictions))
    a = metrics.confusion_matrix(sk_labels, sk_predictions)

# %%
from itertools import chain
predictions_l = [predictions_list[i].tolist() for i in range(len(predictions_list))]
labels_l = [labels_list[i].tolist() for i in range(len(labels_list))]
predictions_l = list(chain.from_iterable(predictions_l))
labels_l = list(chain.from_iterable(labels_l))

# %%

confusion_matrix(labels_l, predictions_l)
print('Classification report for CNN :\n%s\n'%(metrics.classification_report(labels_l, predictions_l)))

# %%
