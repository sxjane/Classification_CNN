#%%
import torch
import torch.nn as nn
import torch.optim as optim 
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np 
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.tensorboard import SummaryWriter

#%%
plt.ion()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
use_gpu = torch.cuda.is_available()
if use_gpu:
    print('Using CUDA')

writer = SummaryWriter()

# %%
data_dir = './data/OCT2017'
TRAIN = 'train'
VAL = 'val'
TEST = 'test'

data_transforms = {
    TRAIN: transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),

    TEST:transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
}
#%%
image_datasets = {
    x: datasets.ImageFolder(
        os.path.join(data_dir, x),
        transform=data_transforms[x]
    )
    for x in [TRAIN, TEST]
}

#%%
dataloaders = {
    TRAIN: torch.utils.data.DataLoader(image_datasets[TRAIN],batch_size=20, shuffle=True,num_workers=4),
    TEST: torch.utils.data.DataLoader(image_datasets[TEST], batch_size=8, shuffle=True, num_workers=4)
}

#%%
dataset_sizes = {
    x: len(image_datasets[x])
    for x in [TRAIN, TEST]
}
for x in [TRAIN, TEST]:
    print('Loaded {} images under {}'.format(dataset_sizes[x],x))

print('Classes')
class_names = image_datasets[TRAIN].classes
print(image_datasets[TRAIN].classes)

#%%
def imshow(inp, title=None):
    inp = inp.cpu().numpy().transpose((1,2,0))
    plt.axis('off')
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
def show_databatch(inputs,classes):
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])

# %%
inputs, classes = next(iter(dataloaders[TRAIN]))
show_databatch(inputs[0:8], classes[0:8])

# %%
def visualize_model(vgg, num_images=6):
    was_trainning=vgg.training
    vgg.train(False)
    vgg.eval()

    images_so_far = 0

    for i, data in enumerate(dataloaders[TEST]):
        inputs, labels = data
        size = inputs.size()[0]

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = vgg(inputs)

        _, preds = torch.max(outputs.data, 1)
        predicted_labels = [preds[j] for j in range(inputs.size()[0])]

        print('Ground Truth:')
        show_databatch(inputs, labels)
        print('Prediction:')
        show_databatch(inputs, predicted_labels)

        del inputs, labels, outputs,preds, predicted_labels
        torch.cuda.empty_cache()

        images_so_far +=size
        if images_so_far >= num_images:
            break

        vgg.train(mode=was_trainning)

# %%
def eval_model(vgg, criterion):
    since = time.time()
    avg_loss = 0
    avg_acc = 0
    loss_test = 0
    acc_test = 0

    test_batches = len(dataloaders[TEST])
    print('Evaluating model')
    print('-' * 10)

    for i, data in enumerate(dataloaders[TEST]):
        if i % 100 == 0:
            print('\rTest batch {}/{}'.format(i, test_batches), end='', flush=True)

        vgg.train(False)
        vgg.eval()
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = vgg(inputs)

        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)

        loss_test += loss.item()
        acc_test += torch.sum(preds == labels).float()

        del inputs, labels, outputs, preds
        torch.cuda.empty_cache()

    avg_loss = loss_test/ dataset_sizes[TEST]
    avg_acc = acc_test/ dataset_sizes[TEST]

    elapsed_time = time.time() - since 
    print()
    print('Evaluation complted in {:.0f}m {:.0f}s'.format(elapsed_time//60, elapsed_time % 60))
    print('Avg loss (test):{:.4f}'.format(avg_loss))
    print('Avg acc (test): {:.4f}'.format(avg_acc))
    print('-' * 10)

vgg16 = models.vgg16(pretrained=True)
for param in vgg16.features.parameters():
    param.require_grad = False
num_features = vgg16.classifier[6].in_features

features = list(vgg16.classifier.children())[:-1]
features.extend([nn.Linear(num_features, len(class_names))])
vgg16.classifier = nn.Sequential(*features)
print(vgg16)
vgg16.to(device)
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(vgg16.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

print('Test before trainning')
eval_model(vgg16, criterion)

# %%
visualize_model(vgg16)

#%%
val_dataloader = [x for i, x in enumerate(dataloaders[TEST]) if i < 30]
train_ratio = 20
val_size = len(val_dataloader) * 8

loss_lists = []
acc_lists = []
iter_lists = []

# %%
def train_model(vgg, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()
    best_model_wts = copy.deepcopy(vgg.state_dict())
    best_acc = 0.0
    
    avg_loss = 0
    avg_acc = 0
    avg_loss_val = 0
    avg_acc_val = 0
    iter_epoch_len = 0

    train_batches = len(dataloaders[TRAIN])
    val_batches = len(val_dataloader)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_features))
        print('-' * 10)

        loss_train = 0
        loss_val = 0
        acc_train = 0
        acc_val = 0
        
        vgg.train(True)

        for i, data in enumerate(dataloaders[TRAIN]):
            
            if i >= train_batches/train_ratio:
                iter_epoch_len +=  i - 1 
                print('iter_len', iter_epoch_len)
                break

            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = vgg(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            acc_iter = torch.sum(preds == labels).float()
            acc_train += acc_iter

            writer.add_scalar('Loss/train', loss.item(), iter_epoch_len + i)
            writer.add_scalar('Accuracy/train', acc_iter/len(inputs), iter_epoch_len + i)

            if i%10 == 0:
                loss_lists.append(loss.item())
                acc_lists.append(acc_iter)
                iter_lists.append(i)
            
            if i%100 == 0:
                print('\rTraining batch {}/{}'.format(i, train_batches/train_ratio, end='', flush=True))
                print('Iteration: {}, Loss:{}, Accuracy:{:.2f}%'.format(i, loss.item(), acc_iter*100/len(inputs)))

            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()
 
        avg_loss = loss_train * train_ratio / dataset_sizes[TRAIN]
        avg_acc = acc_train * train_ratio * 100 / dataset_sizes[TRAIN]
        totoal_iter_epoch = i

        vgg.train(False)
        vgg.eval()

        for i, data in enumerate(val_dataloader):

            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = vgg(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            loss_val += loss.item()
            acc_val += torch.sum(preds == labels.data).float()

            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()

        avg_loss_val = loss_val / val_size
        avg_acc_val = acc_val * 100 / val_size

        print()
        print('Epoch {} result: '.format(epoch))
        print('Avg loss (train): {:.4f}'.format(avg_loss))
        print('Avg acc (train): {:.4f}%'.format(avg_acc))
        print('Avg loss (val):{:.4f}'.format(avg_loss_val))
        print('Avg acc (val):{:.4f}%'.format(avg_acc_val))
        print('-'*10)
        print()

        if avg_acc_val > best_acc:
            best_acc = avg_acc_val
            best_model_wts = copy.deepcopy(vgg.state_dict())

    elapsed_time = time.time() - since
    print()
    print('Training completed in {:.0f}m {:.0f}s'.format(elapsed_time // 60, elapsed_time % 60))
    print('Best acc: {:.4f}'.format(best_acc))

    vgg.load_state_dict(best_model_wts)
    return vgg

vgg16 = train_model(vgg16, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=2)
torch.save(vgg16.state_dict(), 'VGG16_half.pt')

# %%
eval_model(vgg16, criterion)

# %%
visualize_model(vgg16, num_images=32)
