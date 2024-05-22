import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage.feature import corner_harris, peak_local_max
import skimage.io as skio

# Open the image file
image = cv2.imread("sonar3.jpg")
grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Find contours
max_brightness = np.max(grayscale)
_, thresholded = cv2.threshold(grayscale, 0.5 * max_brightness, max_brightness, 0)
contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours
contours = [contour for contour in contours if len(contour) > 7]

# Draw contours
cv2.drawContours(image, contours, -1, (0, 255, 0), 1)

# Draw bounding boxes
contour = np.reshape(np.array(contours[0]), (len(contours[0]), 2))
print(contour)
for contour in contours:
    contour = np.reshape(np.array(contour), (len(contour), 2))
    x_min = np.min(contour, axis=0)[0]
    x_max = np.max(contour, axis=0)[0]
    y_min = np.min(contour, axis=0)[1]
    y_max = np.max(contour, axis=0)[1]
    if (x_max - x_min < 100 and y_max - y_min < 100):
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0,255,0), 3)

# Save the annotated image
cv2.imwrite("Inference.png", image)
