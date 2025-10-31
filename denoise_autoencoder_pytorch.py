import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

#device=torch.device("cuda" if torch.cuda.is_available else "CPU")
#print(f"using device :{device}")
                    

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5),(0.5))

])

train_data=datasets.MNIST(root='./dir',download=True,transform=transform)
test_data=datasets.MNIST(root='./dir',download=False,transform=transform)

train_loader=DataLoader(train_data,batch_size=128,shuffle=True)
test_loader=DataLoader(test_data,batch_size=128,shuffle=False)

#class cnn_MNIST(nn.Module):
#    def __init__(self):
#        super(cnn_MNIST,self).__init__()
#        self.conv1=nn.Conv2d(1,32,3,padding=1)
#        self.pool1=nn.MaxPool2d(2,2)
#
#        self.conv2=nn.Conv2d(32,64,3,padding=1)
#        self.pool2=nn.MaxPool2d(2,2)
#
#        self.fc1=nn.Linear(7*7*64,256)
#        self.fc2=nn.Linear(256,128)
#        self.fc3=nn.Linear(128,10)
#    
#    def forward(self,x):
#        x=self.conv1(x)
#        x=torch.relu(x)
#        x=self.pool1(x)
#
#        x=self.pool2(torch.relu(self.conv2(x)))
#        x=x.view(x.size(0),-1)
#        x=self.fc1(x)
#        x=torch.relu(x)
#        x=self.fc2(x)
#        x=torch.relu(x)
#        x=self.fc3(x)
#        return x

# Architecture
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,16,3,stride=2,padding=1),
            nn.ReLU(True),
            nn.Conv2d(16,32,3,stride=2,padding=1),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
            nn.Tanh()
        )

    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def add_noise(img):
    noisy= torch.randn_like(img) * 0.5
    noisy=img+noisy
    noise=torch.clamp(noisy,-1.,1.)
    return noise

epoch_loss=[]
for epoch in range(20):
    for image,label in train_loader:
        noisy_image=add_noise(image)
        output = model(noisy_image)
        loss=criterion(output,image)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    epoch_loss.append(loss.item())
print(f"epoch_loss:{epoch_loss}")
        
correct=0.0
total=0.0
model.eval()
with torch.no_grad():
    for image,label in test_loader:
        noisy_image=add_noise(image)
        output=model(noisy_image)
        max,predicted=torch.max(output,1)
        correct += (predicted==label).sum().item()
        total+=label.size(0)
    print(f"test accuracy:{(correct/total)*100}")

# Visualization
plt.figure(figsize=(10, 4))
for i in range (6):
    plt.imshow(image[i].squeeze(), cmap = 'gray')
    plt.title('Original Image')
plt.show()

