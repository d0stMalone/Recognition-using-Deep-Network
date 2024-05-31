# Keval Visaria & Chirag Dhoka Jain 
# This Code is for TASK 4

import tensorflow as tf
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
import os
import itertools
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Load MNIST Fashion dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Preprocess data
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

# Data augmentation using ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

datagen.fit(x_train)

# Split training data into training and validation sets
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Define parameter ranges
num_conv_layers = [1, 2,3,4]
num_filters = [16,32, 64,128]
dropout_rates = [ 0.1,0.3, 0.4,0.5]

# Define search strategy
param_combinations = list(itertools.product(num_conv_layers, num_filters, dropout_rates))

# Function to create and train the model
def train_model(num_conv, num_filters, dropout_rate):
    """
    Create and train a convolutional neural network model with specified parameters.
    
    Parameters:
        num_conv (int): Number of convolutional layers.
        num_filters (int): Number of filters in each convolutional layer.
        dropout_rate (float): Dropout rate for regularization.
    
    Returns:
        model: Trained Keras model.
        test_acc (float): Test accuracy of the model.
        training_time (float): Time taken for training.
        history (dict): Training history containing loss and accuracy values.
    """
    model = Sequential()
    
    # Add convolutional layers
    for i in range(num_conv):
        model.add(Conv2D(num_filters, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1) if i == 0 else None))
        model.add(MaxPooling2D((2, 2)))
    
    # Add dropout and dense layers
    model.add(Flatten())
    model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation='softmax'))
    
    # Compile and train the model
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    start_time = time.time()
    history = model.fit(x_train, y_train, epochs=20, batch_size=128, verbose=0, validation_data=(x_val, y_val))
    training_time = time.time() - start_time
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    
    return model, test_acc, training_time, history.history

# Experiment loop
results = []
for num_conv, num_filters, dropout_rate in param_combinations:
    model, test_acc, training_time, history = train_model(num_conv, num_filters, dropout_rate)
    results.append((num_conv, num_filters, dropout_rate, test_acc, training_time, history))
    print(f"Num Conv: {num_conv}, Num Filters: {num_filters}, Dropout Rate: {dropout_rate}, Accuracy: {test_acc:.4f}, Training Time: {training_time:.2f}s")

# Find the best configuration
best_config = max(results, key=lambda x: x[3])
print(f"\nBest Configuration: Num Conv: {best_config[0]}, Num Filters: {best_config[1]}, Dropout Rate: {best_config[2]}, Accuracy: {best_config[3]:.4f}, Training Time: {best_config[4]:.2f}s")

# Plot training and validation accuracy for the best configuration
plt.figure(figsize=(8, 6))
history_dict = best_config[5]
if isinstance(history_dict, dict):
    plt.plot(history_dict['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history_dict:
        plt.plot(history_dict['val_accuracy'], label='Validation Accuracy')
    else:
        print("Warning: 'val_accuracy' not found in history dictionary.")
else:
    print("Warning: History is not a dictionary.")
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
