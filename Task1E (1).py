# Keval Visaria & Chirag Dhoka Jain 
# This Code is for TASK 1 - E
# This script reads the network and runs the model on the first 10 examples in the test set and plots the first 9 digits with prediction.

# Import necessary libraries
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
from Task1AtoD import CnnNetwork
from GaborFilterNetwork import GaborCnnNetwork

# Function to load a pre-trained CNN model from a specified path.
def load_model(model, model_path):
    """
    Initializes an instance of the CNN network, loads the pre-trained model weights,
    and switches the model to evaluation mode.
    """
    model = model  # Initialize an instance of CnnNetwork.
    model.load_state_dict(torch.load(model_path))  # Load the model's state dictionary from the provided path.
    model.eval()  # Switch the model to evaluation mode to deactivate dropout layers and normalize layers.
    return model

# Function to test the model on the MNIST test dataset and visualize predictions.
def run_on_test_set(model, test_loader):
    """
    Iterates through the test dataset to predict and visualize the first 10 examples.
    The predictions, along with the images and true labels, are plotted in a 3x3 grid.
    """
    # Initialize the figure for plotting the first 9 examples in a 3x3 grid.
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    fig.tight_layout(pad=3.0)
    
    for i, (data, target) in enumerate(test_loader):
        if i >= 10:  # Limit to only the first 10 examples for processing and visualization.
            break

        output = model(data)  # Pass the data through the model to obtain predictions.
        pred = output.data.max(1, keepdim=True)[1]  # Extract the predicted label from the model's output.

        # Print detailed information about the prediction for debugging and verification purposes.
        print(f"Example {i}:")
        print(f"Network output: {output.data.numpy()[0].round(2)}")  # Show the network's output rounded for readability.
        print(f"Predicted label: {pred.item()}, Correct label: {target.item()}\n")  # Display the predicted and correct labels.

        # Plot the first 9 examples with their predicted labels.
        if i < 9:
            ax = axes[i // 3, i % 3]  # Determine the subplot for the current image.
            ax.imshow(data.numpy().squeeze(), cmap='gray')  # Visualize the image in grayscale.
            ax.set_title(f"Pred: {pred.item()}")  # Annotate the image with the predicted label.
            ax.axis('off')  # Hide axes for a cleaner look.

    plt.show()  

def main():
    # Path to the saved model.
    # model_path = 'results\CNN_MNIST\model.pth'
    
    # Load the pre-trained model
    # model = load_model(CnnNetwork(), model_path = 'results\CNN_MNIST\model.pth')
    model = load_model(GaborCnnNetwork(), model_path='results\CNN_MNIST\Gabor_model.pth')

    
    # Prepare the MNIST test dataset with appropriate transforms
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    # Run the model on the test set to visualize predictions and model performance
    run_on_test_set(model, test_loader)

if __name__ == "__main__":
    main()