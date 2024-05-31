# Keval Visaria & Chirag Dhoka Jain 
# This Code is for TASK 1 - A to D
# This script creates and trains a convolutional neural network (CNN) using PyTorch for digit classification on the MNIST dataset. 
# It includes network architecture definition, training, and evaluation phases with visualization of training and test loss.

import os
import sys
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from GaborFilterNetwork import GaborCnnNetwork

# Class definition for the neural network
class CnnNetwork(nn.Module):
    
    # CNN with two convolutional layers, dropout, and two fully connected layers for MNIST digit classification.
    
    def __init__(self):
        
        super(CnnNetwork, self).__init__()
        
        # layer1 is a convolutional layer with 10 5x5 filters
        self.layer1 = nn.Conv2d(1, 10, kernel_size=5)
        
        # layer2 is a convolutional layer with 20 5x5 filters
        self.layer2 = nn.Conv2d(10, 20, kernel_size=5)

        # dropout layer with a probability of 0.5
        self.drop_layer = nn.Dropout2d()
        
        # fc1 is a fully connected layer with 50 neurons
        self.fc1 = nn.Linear(320, 50)
        # fc2 is a fully connected layer with 10 neurons
        self.fc2 = nn.Linear(50, 10)
    
    
    #Forward pass of the network.
    def forward(self, x):

        # Apply the first convolutional layer followed by ReLU activation function,
        # then apply max pooling with a 2x2 window
        x = F.relu(F.max_pool2d(self.layer1(x), 2))
        
        # Apply the second convolutional layer, followed by a dropout layer to prevent overfitting,
        # then use ReLU activation function and apply max pooling with a 2x2 window
        x = F.relu(F.max_pool2d(self.drop_layer(self.layer2(x)), 2))
        
        # Flatten the output from the previous layer to a vector
        x = x.view(-1, 320)
        
        # Apply the first fully connected layer followed by ReLU activation function
        x = F.relu(self.fc1(x))
        
        # Apply the second fully connected layer and use the log softmax function on the output
        # Log softmax is often used for classification problems with multiple classes
        x = F.log_softmax(self.fc2(x), dim=1)
        
        return x

    
"""
Trains the network for a single epoch.

The function iterates over the DataLoader, fetching batches of images and their corresponding labels.
For each batch, it performs a forward pass, calculates the loss, performs a backward pass to compute gradients,
and updates the model parameters. Additionally, it logs the loss at regular intervals and saves the model and optimizer states.
"""
def train_network(model, optimizer, epoch, loader, losses, counters, batch_size):

    model.train()  # Set the model to training mode (enables dropout, batch normalization, etc.)

    for idx, (data, target) in enumerate(loader):

        optimizer.zero_grad()  # Clear the gradients of all optimized variables           
        output = model(data)  # Forward pass: compute predicted outputs by passing inputs to the model
        loss = F.nll_loss(output, target)  # Calculate the batch loss using the negative log likelihood loss function
        loss.backward()  # Backward pass: compute gradient of the loss with respect to model parameters
        optimizer.step()  # Perform a single optimization step (parameter update)
        
        # Logging block: executed every 10 batches
        if idx % 10 == 0:

            # Print loss information
            print(f"Epoch {epoch} [{idx * len(data)}/{len(loader.dataset)} ({100. * idx / len(loader)}%)]\tLoss: {loss.item():.6f}")
            losses.append(loss.item())  # Append the current loss to the losses list            
            counters.append(idx * batch_size + (epoch - 1) * len(loader.dataset))  # Update counters with the number of examples seen


"""
Evaluates the network's performance on a dataset using the provided DataLoader.

This function switches the model to evaluation mode, then iterates over the dataset,
computing the model's output for each batch without tracking gradients. It calculates
the total loss and the number of correctly predicted examples to compute the average
loss and the accuracy of the model on the dataset. Finally, it prints the results.

The model's performance is assessed based on the negative log likelihood loss and accuracy,
reflecting how well it predicts the correct classes for the input images.
"""
def evaluate(model, loader, losses):

    model.eval()  # Set the model to evaluation mode (disables dropout, batch normalization, etc.)

    test_loss = 0  # Initialize the total loss to zero
    correct = 0    # Initialize the count of correct predictions to zero

    with torch.no_grad():  # Disable gradient computation for efficiency
        for data, target in loader:
            output = model(data)  # Compute the model's output

            # Accumulate the total loss by adding the loss for the current batch
            test_loss += F.nll_loss(output, target, reduction='sum').item()

            # Compute the predicted class by finding the index with the maximum log-probability
            pred = output.max(1, keepdim=True)[1]

            # Count how many predictions match the true labels
            correct += pred.eq(target.view_as(pred)).sum().item()

    # Calculate the average loss over the entire dataset
    test_loss /= len(loader.dataset)

    # Append the average loss to the losses list for logging or further processing
    losses.append(test_loss)

    # Print the evaluation results including the average loss and the accuracy
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(loader.dataset)} ({100. * correct / len(loader.dataset):.0f}%)\n")


# Orchestrates the training and evaluation of a CNN model for MNIST digit classification.
def main(argv):
    
    # Handle command line arguments for configurable options.
    if len(argv) > 1:
        print(f"Received arguments: {argv[1:]}")

    # Data loading with normalization, and training and test set preparation.
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=True)

    # Visualize initial dataset samples to verify data loading.
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(test_dataset.data[i], cmap="gray", interpolation="none")
        plt.title("{}".format(test_dataset.targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()

    # Initialize the CNN model and the optimizer with given parameters.
    # model = CnnNetwork()
    model = GaborCnnNetwork()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    # Setup training process including loss tracking.
    epochs = 5
    training_losses, training_counters, testing_losses = [], [], []
    testing_counters = [i * len(train_loader.dataset) for i in range(epochs + 1)]

    # Perform initial evaluation, train across epochs, and evaluate after each epoch.
    evaluate(model, test_loader, testing_losses)
    for epoch in range(1, epochs + 1):
        train_network(model, optimizer, epoch, train_loader, training_losses, training_counters, 64)
        evaluate(model, test_loader, testing_losses)
    

    directory = './results/CNN_MNIST'
    if not os.path.exists(directory):  # Check if the directory exists; if not, create it
        os.makedirs(directory)
            
    # Save the current state of the model and optimizer to the directory
    # torch.save(model.state_dict(), f"{directory}/model.pth")
    torch.save(model.state_dict(), f"{directory}/Gabor_model.pth")
    torch.save(optimizer.state_dict(), f"{directory}/Gabor_optimizer.pth")
        
    print(f"Final model and optimizer state saved.")

    # Plot training and testing loss to visualize learning progress.
    plt.figure(figsize=(10, 8))
    plt.plot(training_counters, training_losses, color='green')
    plt.scatter(testing_counters, testing_losses, color='blue')
    plt.legend(['Training Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('Number of training examples seen')
    plt.ylabel('Negative log likelihood loss')
    plt.title('Training and Test Loss')
    plt.show()
 
if __name__ == "__main__":
    main(sys.argv)