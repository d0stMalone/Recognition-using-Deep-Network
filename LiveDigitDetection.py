# Keval Visaria & Chirag Dhoka Jain 
# This Code is for TASK 3 with Extension
# IN this script we use the camera and start a live digit detection
# Extension 6

import cv2
import numpy as np
import torch
from Task1E import load_model 
from Task3 import ModifiedCnnNetwork  
from GaborFilterNetwork import GaborCnnNetwork


# Choose between the original or modified CNN model for digit recognition
# Uncomment the model you wish to use and ensure the correct path is provided
# model = load_model(CnnNetwork(), model_path="path/to/original/model.pth")
# model = load_model(ModifiedCnnNetwork(), model_path="path/to/modified/model.pth")
model = load_model(GaborCnnNetwork(), model_path="results\CNN_MNIST\Gabor_model.pth")

# Initialize a video capture object to start capturing video from the camera.
# Replace '1' with '0' for default camera if an external one isn't connected.
cap = cv2.VideoCapture(1)

while True:
    # Capture a single frame from the ongoing video stream.
    ret, frame = cap.read()

    # Convert the captured frame to grayscale to simplify processing.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply a binary threshold to highlight the digits or objects of interest.
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Detect contours in the thresholded image to find shapes and objects.
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables to find the contour with the largest area.
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    # If a significant contour is found, process it further.
    if max_contour is not None:
        # Generate a mask for the largest contour to isolate it.
        mask = np.zeros_like(thresh)
        cv2.drawContours(mask, [max_contour], -1, 255, -1)

        # Apply the mask to the original frame to focus on the detected object.
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

        # Convert the resulting image to grayscale and resize to match model input.
        masked_gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(masked_gray, (28, 28))
        resized = cv2.bitwise_not(resized)  # Invert colors for better recognition.

        # Display the processed image (cropped digit) for visualization.
        cv2.imshow("Cropped digit", resized)

        # Prepare the processed image for model prediction.
        tensor = torch.from_numpy(resized).unsqueeze(0).float()

        # Predict the digit using the pre-trained model.
        output = model(tensor.unsqueeze(0))  # Add batch dimension.
        digit = output.argmax().item()  # Extract predicted digit.

        # Overlay the prediction on the original frame.
        cv2.putText(frame, str(digit), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    # Display the original video frame with any overlay text.
    cv2.imshow("Video", frame)

    # Break the loop if 'q' key is pressed.
    if cv2.waitKey(1) == ord("q"):
        break

# Release the video capture object and close all OpenCV windows.
cap.release()
cv2.destroyAllWindows()
