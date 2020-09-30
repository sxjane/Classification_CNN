#%%
from torch.utils.data import DataLoader 
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
import torch
import os
import torchvision
from matplotlib import pyplot as plt
import time 
import copy
import torchvision.models as models 
from torch.optim import SGD, lr_scheduler
import torch.nn as nn
import numpy as np 

#%%
#load data from folders 
data_dir = './hymenoptera_data'

mean=[0.485,0.456,0.406]
std=[0.229,0.224,0.225]
data_transforms = {
    'train': T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ]),
    'val': T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])
}

image_datasets = {x: ImageFolder(os.path.join(data_dir, x), 
                                 transform=data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=40, shuffle=True)
               for x in ['train', 'val']}

classes_name = image_datasets['train'].classes

#%%
def imshow(img, titles = None):
    img = img.permute(1,2,0)
    img = img.numpy() * std + mean
    img = np.clip(img, 0, 1)
    plt.figure(figsize=(20,50))
    plt.imshow(img)
    plt.axis('off')
    if titles is not None:
        plt.title(titles, fontdict={'fontsize': 30})
    plt.show()
def imshow_imgs_list(imgs, labels):
    img = torchvision.utils.make_grid(imgs, nrow=5, padding=5)
    titles = [classes_name[x] for x in labels]
    imshow(img, titles)
#%%
writer = SummaryWriter()
if(torch.cuda.is_available()):
    print('use cuda')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# %%
def eval_model(model, criterion):
    model.eval()
    total_acc = 0
    for i, (images, labels) in enumerate(iter(dataloaders['val'])):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        iter_acc = torch.sum(preds == labels).float()
        total_acc += iter_acc 
    avg_acc = (total_acc/len(image_datasets['val'])).item()
    print('Average_accuracy: ', avg_acc)

# %%
def train_model(model, criterion, optimizer, scheduler, num_epochs = 20):
    model.train()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    total_idx = 0

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0.0
        for idx, (images, labels) in enumerate(dataloaders['train']):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            _,preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            acc = torch.sum(preds==labels).float()/len(labels)
            writer.add_scalar('Train/Loss', loss.item(), total_idx)
            writer.add_scalar('Train/Accuracy', acc.item(), total_idx)
            total_idx +=1

            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds==labels)
        running_loss = running_loss / len(image_datasets['train'])
        running_corrects = running_corrects / len(image_datasets['train'])
        writer.add_scalar('Train/Avg_loss', running_loss, epoch)
        writer.add_scalar('Train/Avg_corrects', running_corrects, epoch)          
        scheduler.step()
        
        model.eval()
        val_images, val_labels = next(iter(dataloaders['val']))
        val_images = val_images.to(device)
        val_labels = val_labels.to(device)
        outputs = model(val_images)
        _,preds = torch.max(outputs, 1)
        val_loss = criterion(outputs, val_labels)
        val_acc = torch.sum(preds == val_labels).float()/len(val_labels)
        writer.add_scalar('Val/Accuracy', val_acc, epoch)
        writer.add_scalar('Val/Loss', val_loss, epoch)

        if val_acc > best_acc:
            best_model_wts = copy.deepcopy(model.state_dict())
            
    model.load_state_dict(best_model_wts)
       
    return model

# %%
resnet = models.resnet50(pretrained=True)
for param in resnet.parameters():
    param.requires_grad = False
in_features = resnet.fc.in_features
out_features = len(classes_name)
fc_layer = nn.Linear(in_features,out_features)
resnet.fc = nn.Linear(in_features,out_features)
resnet.to(device)

model = resnet
criterion = torch.nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(),lr=0.01,momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7,gamma=0.1)
print('Test before training')
eval_model(model, criterion)
result = train_model(model, criterion, optimizer, scheduler, num_epochs=50)

#%%
torch.save(model.state_dict(), os.path.join(data_dir, 'best_model.pth'))

#%%
print('Test after trainning')
eval_model(model, criterion)

# %%
demo_images, demo_labels = next(iter(dataloaders['val']))
model.eval()
demo_images = demo_images.to(device)
outputs = result(demo_images)
_,predictions = torch.max(outputs, 1)
imshow_imgs_list(demo_images.cpu()[0:20], predictions.cpu().numpy()[0:20])
print('Ground Truth', demo_labels[0:20])
print('Prediction', predictions.cpu()[0:20])

# %%
test_dataset = ImageFolder(os.path.join(data_dir, 'test'),transform=data_transforms['val'])
test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=True)

# %%
for demo_images, demo_labels in iter(test_dataloader):
    model.eval()
    demo_images = demo_images.to(device)
    outputs = result(demo_images)
    _,predictions = torch.max(outputs, 1)
    imshow_imgs_list(demo_images.cpu(), predictions.cpu().numpy())
    print('Ground Truth', demo_labels)
    print('Prediction', predictions.cpu())

# %%
