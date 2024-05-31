# Keval Visaria & Chirag Dhoka Jain 
# This Code is for TASK 3 with Extension

import os
import sys
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image
import math
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from Task1AtoD import CnnNetwork, train_network, evaluate
import Task3
from Task1E import load_model
import Task1F
from GaborFilterNetwork import GaborCnnNetwork

prediction_to_greek_letter = {
    0: "Alpha",
    1: "Beta",
    2: "Delta",
    3: "Eta",
    4: "Gamma",
    5: "Theta"
}

def evaluate_on_MNIST(model, data_loader, max_images):
    """
    Evaluates the model on the MNIST dataset and plots a few example images with their predicted and true labels.
    """
    model.eval()
    all_preds, all_labels = [], []

    plt.figure(figsize=(10, 10))

    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            if i >= max_images:
                break
            
            outputs = model(images)
            pred_label = outputs.max(1, keepdim=True)[1]

            all_preds.extend(pred_label.view_as(labels).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Convert labels and predictions to Greek letters using the dictionary
            # Note: Ensure your dictionary maps tensor item to Greek letter as string.
            greek_letter_pred = prediction_to_greek_letter.get(pred_label.item(), "Unknown")
            greek_letter_true = prediction_to_greek_letter.get(labels.item(), "Unknown")

            plt.subplot((max_images + 1) // 3, 3, i + 1)
            plt.imshow(images[0].squeeze().numpy(), cmap="gray")
            # Use the mapped Greek letters directly without calling .item()
            plt.title(f"Pred: {greek_letter_pred}, True: {greek_letter_true}")
            # plt.title(f"Pred: {pred_label.item()}, True: {labels.item()}")
            plt.axis('off')

    plt.show()


class TransformGreekLetters:
    def __call__(self, img):
        img = transforms.functional.rgb_to_grayscale(img)
        img = transforms.functional.affine(img, angle=0, translate=(0, 0), scale=36/128, shear=0)
        img = transforms.functional.center_crop(img, output_size=(28, 28))
        return transforms.functional.invert(img)



def main(args):

    print("Select the network to use:")
    print("1 - Original CnnNetwork")
    print("2 - ModifiedCnnNetwork")
    choice = input("Enter your choice (1 or 2): ")

    if choice == '1':
        # model = CnnNetwork()
        print("Using Original CnnNetwork.")
        model = load_model(CnnNetwork(), model_path='results/CNN_MNIST/model.pth')
        # epochs = 27  # Number of epochs to train
    
    elif choice == '2':
        # model = Task3.ModifiedCnnNetwork()
        print("Using ModifiedCnnNetwork.")
        model = load_model(Task3.ModifiedCnnNetwork(), model_path='results/CNN_MNIST/new_model.pth')
        # epochs = 20  # Number of epochs to train

    else:
        print("Invalid choice, defaulting to Original CnnNetwork.")
        # model = CnnNetwork()
        model = load_model(CnnNetwork(), model_path='results/CNN_MNIST/model.pth')
        # epochs = 27  # Number of epochs to train

    # Freeze all model parameters to prevent them from being updated during training
    for param in model.parameters():
        param.requires_grad = False

    # Modify the final layer to adapt to the number of Greek letter classes
    model.fc2 = nn.Linear(model.fc2.in_features, 6) 

    # Define transformations and load the custom Greek letter dataset
    greek_dataset_path = "greek_train_extension"
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

    epochs = 500

    # Train the model for the specified number of epochs
    for epoch in range(1, epochs + 1):
        train_network(
            model, optimizer, epoch, train_loader,
            training_losses, training_steps, batch_size=54  # Batch size used during training
        )

    # Evaluate the model on the test dataset and plot predictions
    evaluate_on_MNIST(model, test_loader, 12)  # Visualize model predictions on 12 images

    # Further evaluation on handwritten Greek images
    image_dir = 'handwritten_greek_extension'  # Directory containing handwritten Greek images
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
    
    plt.figure()
    plt.plot(training_steps, training_losses, color="green")
    plt.title("Training Loss Over Time")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.show()

if __name__ == "__main__":
    main(sys.argv)