# Keval Visaria & Chirag Dhoka Jain 
# This Code is for Extension 4
# This script uses pre-trained VGG16 model and the CIFAIR-10 dataset to evaluate and visualize the first 3 convolution layers.
# And, also applies a filter on the 6th image of the dataset

import torchvision.models as models
import matplotlib.pyplot as plt
import cv2
import numpy as np
from torchvision import transforms, datasets

# Load a pre-trained VGG16 model from torchvision and set it to evaluation mode.
# This disables layers like dropout and batch normalization during inference.
model = models.vgg16(pretrained=True)
model.eval()

def visualize_filters(layer_weights, title):
    """
    Visualizes the filters of a convolutional layer.
    """
    # Calculate the number of filter rows and columns to display
    n_filters = layer_weights.shape[0]
    n_col = int(np.ceil(np.sqrt(n_filters)))  # Columns based on square root of filter count
    n_row = n_col  # Equal number of rows and columns for a square layout
    
    # Setup subplot grid and title
    fig, axes = plt.subplots(n_row, n_col, figsize=(12, 12))
    fig.suptitle(title, fontsize=16)
    
    # Loop through all the subplots and fill them with filter visualizations
    for i, ax in enumerate(axes.flat):
        if i < n_filters:  # Check to avoid index error if filters < grid size
            # Display the ith filter using 'viridis' colormap
            ax.imshow(layer_weights[i, 0], cmap='viridis')
            ax.axis('off')  # Hide axes for clarity
    
    plt.subplots_adjust(wspace=0.1, hspace=0.2)  # Adjust spacing between plots
    plt.show()

# Indices of the first three convolutional layers in VGG16 architecture.
# These indices correspond to the model's 'features' attribute sequence.
layers_to_visualize = [0, 2, 5]

# Visualize filters in the selected layers.
for i, layer_index in enumerate(layers_to_visualize):
    # Extract and prepare the filter weights from the model
    weights = model.features[layer_index].weight.data.cpu().numpy()
    # Visualize using the helper function
    visualize_filters(weights, f'Layer {i+1} Filters')

# Load and preprocess an image from CIFAR10 for model input.
# VGG16 requires 224x224 images, so we resize the CIFAR10 images.
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 pixels
    transforms.ToTensor(),  # Convert images to PyTorch tensors
])
dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
image, _ = dataset[5]  # Extract the sixth image in the dataset (index 5)

# Convert the tensor image to a format compatible with OpenCV for filtering.
image_np = image.permute(1, 2, 0).numpy()  # Change tensor layout to HxWxC
image_np = cv2.cvtColor(np.float32(image_np), cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

# Select the first filter from the first layer to apply to the image.
first_layer_weights = model.features[0].weight.data.cpu().numpy()
filter_idx = 0  # Index of the first filter
filter_kernel = first_layer_weights[filter_idx, 0]  # Extract the kernel

# Apply the filter to the image using OpenCV's filter2D function.
filtered_image = cv2.filter2D(src=image_np, ddepth=-1, kernel=filter_kernel)

# Visualize the original and the filtered image side by side for comparison.
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(np.clip(image_np, 0, 1))  # Clip values to [0, 1] range for display
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(filtered_image, cmap='gray')  # Display filtered image in grayscale
plt.title('Filtered Image - First Layer, First Filter')
plt.axis('off')

plt.show()
