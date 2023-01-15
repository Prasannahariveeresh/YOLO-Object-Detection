import numpy as np
import cv2

import matplotlib.pyplot as plt

LABELS = open('labels.txt', 'r').read().split('\n')

# Load YOLO model
model = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

# Read image
img = cv2.imread("image.jpg")
height, width = img.shape[:2]

# Create input blob
blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)

# Set input blob for the model
model.setInput(blob)

# Run forward pass
output_layers = model.getUnconnectedOutLayersNames()
layerOutputs = model.forward(output_layers)

# Initialize lists of detected bounding boxes, confidences and class IDs
boxes = []
confidences = []
class_ids = []

# Loop over each output layer
for output in layerOutputs:
    # Loop over each detection
    for detection in output:
        # Extract the class ID and confidence (i.e., probability) of the current object detection
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        # Filter out weak predictions
        if confidence > 0.5:
            # Extract the bounding box coordinates
            box = detection[0:4] * np.array([width, height, width, height])
            (centerX, centerY, width, height) = box.astype("int")

            # Use the center (x, y)-coordinates to derive the top and
            # and left corner of the bounding box
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))

            # Update list of bounding boxes, confidences, and class IDs
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            class_ids.append(class_id)
            

# Apply non-maximum suppression
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

# Draw boxes
for i in indices:
    box = boxes[i]
    x, y, w, h = box
    cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
    text = "{}: {:.4f}".format(LABELS[class_ids[i]], confidences[i])
    cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

plt.imshow(img)
plt.show()
