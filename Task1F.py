# Keval Visaria & Chirag Dhoka Jain 
# This Code is for TASK 1 - F
# This script reads the network and runs the model on the handwritten inputs.

# Import necessary libraries
import os
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from Task1AtoD import CnnNetwork  # Import the CNN network class defined in Task1AtoD
from Task1E import load_model
from GaborFilterNetwork import GaborCnnNetwork

# Function to load and preprocess an image
def load_preprocess_image(image_path):
    """
    Loads an image from the specified path, converts it to grayscale, inverts its colors,
    resizes it to 28x28 (the input size required by the model), and normalizes it
    according to the MNIST dataset's parameters. Finally, converts the image to a PyTorch tensor.
    """
    img = Image.open(image_path).convert('L')  # Open and convert the image to grayscale.
    img = ImageOps.invert(img)  # Invert the image colors to match the MNIST dataset's format.
    
    # Define a series of transformations to prepare the image for the model
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # Resize the image to 28x28 pixels, as expected by the model.
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor.
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize the tensor using the mean and std of the MNIST dataset.
    ])
    
    img_tensor = transform(img).unsqueeze(0)  # Apply transformations and add a batch dimension.
    return img_tensor

# Function to classify an image tensor using the model
def classify_image(model, image_tensor):
    """
    Classifies an image represented as a PyTorch tensor using the provided model.
    The function performs a forward pass through the model without computing gradients,
    and returns the predicted class for the image.
    """
    model.eval()  # Ensure the model is in evaluation mode.
    
    with torch.no_grad():  # Temporarily disable gradient computation.
        output = model(image_tensor)  # Compute the model's output for the input tensor.
        pred = output.argmax(dim=1, keepdim=True)  # Determine the class with the highest score.
        return pred.item()  # Return the predicted class label as an integer.


def main():
    
    # Load the pre-trained model.
    
    # model = load_model(CnnNetwork(), model_path='results\CNN_MNIST\model.pth')
    model = load_model(GaborCnnNetwork(), model_path='results\CNN_MNIST\Gabor_model.pth')
    
    # Specify the directory containing the digit images to classify.
    image_dir = 'C:/Users/visar/Desktop/OneDrive - Northeastern University/PRCV/RDN/digits/'

    # List and sort all JPG images in the directory.
    digit_images = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    
    # Setup the plot for displaying images and their predictions.
    plt.figure(figsize=(10, 4))
    for i, image_name in enumerate(digit_images):
        image_path = os.path.join(image_dir, image_name)  # Construct the full path to the image.
        image_tensor = load_preprocess_image(image_path)  # Preprocess the image and convert it to a tensor.
        prediction = classify_image(model, image_tensor)  # Predict the digit using the model.

        img = Image.open(image_path)  # Load the image again for plotting.
        plt.subplot(2, 5, i+1)  # Position each image in a grid.
        plt.imshow(img, cmap='gray')
        plt.title(f'Pred: {prediction}')  # Annotate the image with the predicted digit.
        plt.axis('off')  # Hide axis ticks and labels for a cleaner look.

    plt.tight_layout()
    plt.show()  # Display the plot with the images and their predicted labels.

if __name__ == "__main__":
    main()  # Execute the main function.

