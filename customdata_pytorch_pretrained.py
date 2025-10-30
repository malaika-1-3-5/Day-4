import torch
import torch.nn as nn
import torch.optim as optim
#import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available else)
print(f"Using device: {device}")
#These 2 lines are written to use the GPU(if available in the system) to run deep learning

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

train_data = datasets.FashionMNIST(root='./dir', train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root='./dir', train=False, download=False, transform=transform)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)
class cnn_Fashionmnist(nn.Module):
    def __init__(self):
        super(cnn_Fashionmnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=1) # input channels = 1 for FashionMNIST
        # As you move further in the convolution, the number of layers must increase but the size of the filter must reduce
        # 3 conclusions --- (1) no. of channels = no. of filters; (2) as the depth increases, the number of filters increases; (3) as the depth increases, the size of filters decreases
        # kernel_size decreases as depth increases because as the depth increases, the image size decreases
        self.pool1 = nn.MaxPool2d(2, 2) #Usually take 2x2 filter size

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # maxpool reduces image, channels increase as set by conv
        self.pool2 = nn.MaxPool2d(2, 2)
        # If we use the pooling layer (2, 2) the size is halved, the number of channels does not change
        # 28x28x1 -> 14x14x32
        # 14x14x64 -> 7x7x64; 7x7 image size, 64 channels

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(6*6*64, 256) #Number of inputs here is the number of pixels after conv/pool
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)

        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.flatten(1) 
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = torch.relu(x)

        x = self.fc2(x)
        x = torch.relu(x)

        x = self.fc3(x)
        x = torch.relu(x)

        x = self.fc4(x)
        x = torch.relu(x)

        x = self.fc5(x)
        return x
    
model = cnn_Fashionmnist()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
for epoch in range(20):
    for image, label in train_loader:
        image, label = image.to(device), label.to(device)
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