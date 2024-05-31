# Keval Visaria & Chirag Dhoka Jain 
# This Code is for Extension 5
# In this script we create a gabor filter kernel and apply it on the first layer of CnnNetwork, this new model is called GaborCnnNetwork
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import math

def gabor_kernel(frequency, theta, sigma_x, sigma_y, n_stds=3, grid_size=5):
    """
    Generates a Gabor kernel with the specified parameters.
    """
    x_max = y_max = grid_size // 2
    x_grid, y_grid = torch.meshgrid(torch.arange(-x_max, x_max + 1), torch.arange(-y_max, y_max + 1))
    x_theta = x_grid * math.cos(theta) + y_grid * math.sin(theta)
    y_theta = -x_grid * math.sin(theta) + y_grid * math.cos(theta)
    kernel = torch.exp(-0.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2)) * torch.cos(2 * math.pi * frequency * x_theta)
    return kernel / kernel.sum()

class GaborCnnNetwork(nn.Module):
    def __init__(self):
        super(GaborCnnNetwork, self).__init__()
        self.layer1 = nn.Conv2d(1, 10, kernel_size=5, padding=2, bias=False)
        for i, filter in enumerate(self.layer1.weight):
            theta = np.pi * i / 10
            frequency = 0.4
            sigma_x = sigma_y = 2.0
            filter.data[:] = gabor_kernel(frequency, theta, sigma_x, sigma_y, grid_size=5)
        self.layer1.weight.requires_grad = False
        
        self.layer2 = nn.Conv2d(10, 20, kernel_size=5)
        self.drop_layer = nn.Dropout2d()
        self.fc1 = nn.Linear(5*5*20, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.layer1(x), 2))
        x = F.relu(F.max_pool2d(self.drop_layer(self.layer2(x)), 2))
        x = x.view(-1, 5*5*20)
        x = F.relu(self.fc1(x))
        x = self.drop_layer(x)
        x = F.log_softmax(self.fc2(x), dim=1)
        return x

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def main():
    # Initialize the Gabor CNN model and set the computation device
    model = GaborCnnNetwork()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define the optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    # Prepare the MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Train the model
    for epoch in range(1, 11):  # 10 epochs
        train(model, device, train_loader, optimizer, epoch)

    # Save the trained model
    torch.save(model.state_dict(), "Gabor_model.pth")
    print("Model saved!")

if __name__ == "__main__":
    main()
