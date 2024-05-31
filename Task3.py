# Keval Visaria & Chirag Dhoka Jain 
# This Code is for TASK 3

# Import required libraries
import os
import sys
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image
from Task1AtoD import CnnNetwork, train_network  # Utilizing the previously defined CNN model and training function
import Task1F  # Assuming this contains utility functions for image preprocessing and classification
from Task1E import load_model  # Loading model utility from Task 1E
import math

class ModifiedCnnNetwork(CnnNetwork):
    def __init__(self):
        super(ModifiedCnnNetwork, self).__init__()
        # Overriding the convolutional layers
        self.layer1 = nn.Conv2d(1, 16, kernel_size=3)  # Smaller filter and more filters
        self.layer2 = nn.Conv2d(16, 32, kernel_size=5)  # More filters
        
        # Overriding/Adding dropout layers
        self.drop_layer = nn.Dropout2d(0.25)  # Adjusted dropout rate for conv layers
        self.drop_layer2 = nn.Dropout2d(0.5)  # Additional dropout layer after fully connected layer
        
        # Overriding the fully connected layers
        self.fc1 = nn.Linear(512, 100)  # Adjusted for the new flattening size and neuron count
        self.fc2 = nn.Linear(100, 10)
    
    def forward(self, x):
        # Using Leaky ReLU instead of ReLU and adjusting forward pass accordingly
        x = F.leaky_relu(F.max_pool2d(self.layer1(x), 2))
        x = F.leaky_relu(F.max_pool2d(self.drop_layer(self.layer2(x)), 2))
        x = x.view(-1, 512)  # Adjusting flattening size
        x = F.leaky_relu(self.fc1(x))
        x = self.drop_layer2(x)  # Applying additional dropout here
        x = F.log_softmax(self.fc2(x), dim=1)
        return x
    
# Mapping from numerical predictions to Greek letters for interpretability
prediction_to_greek_letter = {
    0: "Alpha",
    1: "Beta",
    2: "Gamma",
}

def evaluate_on_MNIST(model, data_loader, max_images):
    """
    Evaluates the model on the MNIST dataset (or a similar structure dataset)
    and plots example images with their predicted and true labels mapped to Greek letters.
    """
    plt.figure(figsize=(10, 10))  # Set figure size for the plots

    with torch.no_grad():  # Disable gradient computations for evaluation
        for i, (images, labels) in enumerate(data_loader):
            if i >= max_images:  # Limit the number of images to display
                break
            
            outputs = model(images)  # Get model predictions
            pred_label = outputs.max(1, keepdim=True)[1]  # Extract predicted labels

            # Map numerical labels to Greek letters
            greek_letter_pred = prediction_to_greek_letter.get(pred_label.item(), "Unknown")
            greek_letter_true = prediction_to_greek_letter.get(labels.item(), "Unknown")

            # Plot each image with its predicted and true labels
            plt.subplot((max_images + 1) // 3, 3, i + 1)
            plt.imshow(images[0].squeeze().numpy(), cmap="gray")
            plt.title(f"Pred: {greek_letter_pred}, True: {greek_letter_true}")
            plt.axis('off')

    plt.show()  # Display the plotted figures

# Custom transformation class for Greek letters dataset preprocessing
class TransformGreekLetters:
    def __call__(self, img):
        """
        Applies a series of transformations to images to match MNIST format,
        including grayscale conversion, resizing, cropping, and inversion.
        """
        img = transforms.functional.rgb_to_grayscale(img)  # Convert RGB to grayscale
        img = transforms.functional.affine(img, angle=0, translate=(0, 0), scale=36/128, shear=0)  # Resize
        img = transforms.functional.center_crop(img, output_size=(28, 28))  # Crop to 28x28 pixels
        return transforms.functional.invert(img)  # Invert colors to match MNIST

def main(args):
    """
    Main function to load a pre-trained CNN model, adapt it to classify Greek letters,
    and evaluate its performance on a custom dataset and MNIST.
    """
    print("Select the network to use:")
    print("1 - Original CnnNetwork")
    print("2 - ModifiedCnnNetwork")
    choice = input("Enter your choice (1 or 2): ")

    if choice == '1':
        # model = CnnNetwork()
        print("Using Original CnnNetwork.")
        model = load_model(CnnNetwork(), model_path='results\CNN_MNIST\model.pth')
        epochs = 27  # Number of epochs to train
    
    elif choice == '2':
        model = ModifiedCnnNetwork()
        print("Using ModifiedCnnNetwork.")
        model = load_model(ModifiedCnnNetwork(), model_path='results/CNN_MNIST/new_model.pth')
        epochs = 20  # Number of epochs to train

    else:
        print("Invalid choice, defaulting to Original CnnNetwork.")
        # model = CnnNetwork()
        model = load_model(CnnNetwork(), model_path='results/CNN_MNIST/model.pth')
        epochs = 27  # Number of epochs to train

   

    # Freeze all model parameters to prevent them from being updated during training
    for param in model.parameters():
        param.requires_grad = False

    # Modify the final layer to adapt to the number of Greek letter classes
    model.fc2 = nn.Linear(model.fc2.in_features, 4)  # Assuming 4 classes including 'unknown'

    # Define transformations and load the custom Greek letter dataset
    greek_dataset_path = "greek_train"
    greek_transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        TransformGreekLetters(),  # Apply custom preprocessing defined above
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize using MNIST dataset parameters
    ])
    
    # Create training and testing datasets and dataloaders
    train_dataset = datasets.ImageFolder(root=greek_dataset_path, transform=greek_transform)
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    test_dataset = datasets.ImageFolder(root=greek_dataset_path, transform=greek_transform)
    test_loader = DataLoader(test_dataset, batch_size=1)

    # Set up the optimizer for training
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.7)

    # Training the network
    training_losses = []  # Track training losses
    training_steps = []  # Track steps for plotting purposes

    # Train the model for the specified number of epochs
    for epoch in range(1, epochs + 1):
        train_network(
            model, optimizer, epoch, train_loader,
            training_losses, training_steps, batch_size=54  # Batch size used during training
        )

    # Evaluate the model on the test dataset and plot predictions
    evaluate_on_MNIST(model, test_loader, 12)  # Visualize model predictions on 12 images

    # Further evaluation on handwritten Greek images
    image_dir = 'handwritten_greek'  # Directory containing handwritten Greek images
    digit_images = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    
    # Plotting predictions for handwritten Greek images
    num_images = len(digit_images)
    cols = 5  # Number of columns in the plot
    rows = math.ceil(num_images / cols)  # Calculate rows needed
    plt.figure(figsize=(10, 4))  # Set figure size
    for i, image_name in enumerate(digit_images):
        image_path = os.path.join(image_dir, image_name)
        image_tensor = Task1F.load_preprocess_image(image_path)  # Preprocess image
        prediction = Task1F.classify_image(model, image_tensor)  # Classify image
        
        greek_letter = prediction_to_greek_letter.get(prediction, "Unknown")  # Get Greek letter

        img = Image.open(image_path)  # Open image for plotting
        plt.subplot(rows, cols, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(f'Pred: {greek_letter}')  # Display prediction
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()  # Show the plotted images
    
    # Finally, plot the training loss over time
    plt.figure()
    plt.plot(training_steps, training_losses, color="green")  # Plot training losses
    plt.title("Training Loss Over Time")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.show()

if __name__ == "__main__":
    main(sys.argv)
