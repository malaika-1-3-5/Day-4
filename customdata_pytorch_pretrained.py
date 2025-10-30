import torch
import torch.nn as nn
import torch.optim as optim
#import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

#device = torch.device("cuda" if torch.cuda.is_available else)
#print(f"Using device: {device}")
#These 2 lines are written to use the GPU(if available in the system) to run deep learning

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), #Because of coloured images
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(128) #Every image is of different dimensions from each customer/user, so these images will be cropped to 128x128
])

#train_data = datasets.FashionMNIST(root='./dir', train=True, download=True, transform=transform)
#test_data = datasets.FashionMNIST(root='./dir', train=False, download=False, transform=transform)
#The above 2 lines of code are removed because we are not downloading any dataset

#train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
#test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

data_dir = 'AntsBees'
image_datasets = {}
image_datasets['train'] =  datasets.ImageFolder(os.path.join(data_dir, 'train'), transform = transform)

image_datasets['val'] =  datasets.ImageFolder(os.path.join(data_dir, 'val'), transform = transform)

train_loader = DataLoader(image_datasets['train'], batch_size=128, shuffle=True)
test_loader = DataLoader(image_datasets['val'], batch_size=128, shuffle=False)

# load pre-trained models
model = models.resnet18(pretrained = True) #weights='IMAGENET1K_V1
print(model)

# Freeze all the layers initially
for param in model.parameters():
    param.requires_grad = False #No need to calculate gradient for this beuase gradient means updating the weights

# Replace final fully connected layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(image_datasets['train']))

#(fc): Linear(in_features=512, out_features=1000, bias=True)

# Train
for epoch in range(20):
    for image, label in train_loader:
        #image, label = image.to(device), label.to(device)
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
#    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

correct = 0.0
total = 0
model.eval()
with torch.no_grad():
    for images, label in test_loader:
        output = model(image)
        max, predicted = torch.max(output, 1)
        correct += (predicted==label).sum().item()
        total += label.size(0)
    print(f"Test accuacy: {(correct/total)*100}%")

print("####")