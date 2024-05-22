import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
coco = YOLO('yolov8s.pt')
oi7 = YOLO('yolov8s-oiv7.pt')

# Open the image file
image = cv2.imread("boat.jpg")

# Run YOLOv8 inference on the image
coco_results = coco.predict(image, conf=0.2)

# Visualize the results on the image
annotated_image = coco_results[0].plot()

# Run YOLOv8 inference on the image
oi7_results = oi7.predict(annotated_image, conf=0.2)

# Visualize the results on the image
annotated_image_2 = oi7_results[0].plot()

# Save the annotated image
cv2.imwrite("YOLOv8Inference.png", annotated_image_2)