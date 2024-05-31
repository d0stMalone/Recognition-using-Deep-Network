# Keval Visaria & Chirag Dhoka Jain 
# This Code is for TASK 2
# This script load a pre-trained CNN model, visualizes the weights of the first convolutional layer, 
# and applies these weights as filters to an MNIST sample image to observe their effects

# Import necessary libraries
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
from Task1AtoD import CnnNetwork  # Custom CNN model from Task 1A to D
from Task1E import load_model  # Function to load a pre-trained model from Task 1E

def visualize_model_weights(conv_weights):
    """
    Visualizes the weights/filters of a convolutional layer in a grid format.
    """
    # Set up figure dimensions
    fig = plt.figure(figsize=(9, 8))
    num_filters = conv_weights.shape[0]  # Number of filters in the convolutional layer
    columns, rows = 4, 3  # Grid size for displaying filters

    # Loop through all the filters and plot each one
    for i in range(1, num_filters + 1):
        ax = fig.add_subplot(rows, columns, i)
        ax.title.set_text(f'Filter {i-1}')
        # Display the i-th filter. Note: The indexing i-1 is used to match filter numbering starting from 0.
        plt.imshow(conv_weights[i-1][0].squeeze(), cmap='gray')
        plt.axis('off')  # Hide axes for clarity
    plt.show()

def apply_and_visualize_filters(image_tensor, conv_weights):
    """
    Applies each filter in the conv_weights to the image_tensor, and visualizes
    both the filter and the resulting filtered image side by side.
    """
    # Prepare figure for visualizing the effects of filters on the input image
    fig = plt.figure(figsize=(9, 8))
    columns, rows = 4, 5  # Adjust grid size based on the number of filters

    # Convert the tensor image to a numpy array for processing with cv2
    image_array = image_tensor.squeeze().numpy()

    for i, weight in enumerate(conv_weights):
        # Extract the i-th filter as a numpy array
        filter_kernel = weight[0].numpy()
        # Apply the filter to the image using OpenCV's filter2D function
        filtered_image = cv2.filter2D(image_array, -1, filter_kernel)

        # Visualize the filter itself
        plt.subplot(rows, columns, 2*i+1)
        plt.imshow(filter_kernel, cmap='gray')
        plt.axis('off')

        # Visualize the effect of the filter on the image
        plt.subplot(rows, columns, 2*i+2)
        plt.imshow(filtered_image, cmap='gray')
        plt.axis('off')

    plt.show()

def main():

    # Load the pre-trained CNN model
    model = load_model(CnnNetwork(), model_path='results\CNN_MNIST\model.pth')

    # Extract weights for the first convolutional layer from the model
    first_layer_weights = model.layer1.weight.detach()

    # Visualize the weights of the first convolutional layer
    visualize_model_weights(first_layer_weights)

    # Load an example image from the MNIST dataset
    mnist_data = datasets.MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
    sample_image, _ = mnist_data[0]  # Get the first sample image and its label

    # Apply the first convolutional layer's filters to the image and visualize the effects
    apply_and_visualize_filters(sample_image, first_layer_weights)

if __name__ == '__main__':
    main()
