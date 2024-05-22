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
ret, thresh = cv2.threshold(grayscale, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours
cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

# Save the annotated image
cv2.imwrite("Inference.png", image)
