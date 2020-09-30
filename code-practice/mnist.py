#%%
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
dataset = MNIST('./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=15)
# %%
from torch import nn

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
    nn.LogSoftmax(dim=1)
)
loss = nn.NLLLoss()

print(model)

# %%
from torch import optim
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

running_loss = 0
num_items = 0
for i in range(3):
    for img, lbl in iter(dataloader):
        res = model(img)
        lss = loss(res, lbl)
        lss.backward()
        optimizer.step()
        running_loss += lss.item()
        num_items += 1
    print("Epoch %d %f" % (i, running_loss / num_items))

# %%
img, lbl = next(iter(dataloader))
res = model(img)
lss = loss(res, lbl)
print(lss.item())

# %%
