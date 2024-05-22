import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
coco = YOLO('yolov8s.pt')
oi7 = YOLO('yolov8s-oiv7.pt')

# Open the video file
cap = cv2.VideoCapture(0)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        coco_results = coco.predict(frame, conf=0.5)

        # Visualize the results on the frame
        annotated_frame = coco_results[0].plot()

        # Run YOLOv8 inference on the frame
        oi7_results = oi7.predict(annotated_frame, conf=0.5)

        # Visualize the results on the frame
        annotated_frame_2 = oi7_results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame_2)

        # Break the loop, release the video capture object, and close the display window if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            break
