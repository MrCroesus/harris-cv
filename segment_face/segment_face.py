import cv2
import face_recognition
from PIL import Image, ImageDraw
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
import matplotlib.pyplot as plt

def show_mask(mask, ax):
    color = np.array([1, 1, 1, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def feature_to_point(feature):
    return np.round(np.average(feature, axis=0))

# webcam input
cv2.namedWindow("Segment Face")
cap = cv2.VideoCapture(0)

# matplotlib interactive mode
plt.ion()

# segment anything model
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
predictor = SamPredictor(sam)

while cap.isOpened():
    # get frames
    ret, frame = cap.read()
    if not ret: break
    
    # find facial features
    faces = face_recognition.face_landmarks(frame)
    
    pil_frame = Image.fromarray(frame)
    for face in faces:
        # virtual makeup
        draw_frame = ImageDraw.Draw(pil_frame, 'RGBA')

        # chin
        draw_frame.polygon(face['chin'], fill=(0, 0, 0, 128))
        draw_frame.line(face['chin'], fill=(0, 215, 255, 128), width=10)

        # eyebrows
        draw_frame.polygon(face['left_eyebrow'], fill=(0, 75, 150, 255))
        draw_frame.line(face['left_eyebrow'], fill=(0, 0, 0, 255), width=10)
        draw_frame.polygon(face['right_eyebrow'], fill=(0, 75, 150, 255))
        draw_frame.line(face['right_eyebrow'], fill=(0, 0, 0, 255), width=10)

        # eyes
        draw_frame.polygon(face['left_eye'], fill=(255, 0, 0, 255))
        draw_frame.line(face['left_eye'], fill=(255, 255, 255, 255), width=5)
        draw_frame.polygon(face['right_eye'], fill=(255, 0, 0, 255))
        draw_frame.line(face['right_eye'], fill=(255, 255, 255, 255), width=5)

        # nose
        draw_frame.polygon(face['nose_bridge'], fill=(0, 255, 255, 255))
        draw_frame.line(face['nose_bridge'], fill=(192, 192, 192, 255), width=5)
        draw_frame.polygon(face['nose_tip'], fill=(255, 255, 0, 255))
        draw_frame.line(face['nose_tip'], fill=(0, 255, 0, 255), width=5)

        #lips
        draw_frame.polygon(face['top_lip'], fill=(0, 0, 255, 255))
        draw_frame.line(face['top_lip'], fill=(255, 0, 255, 255), width=5)
        draw_frame.polygon(face['bottom_lip'], fill=(0, 0, 255, 255))
        draw_frame.line(face['bottom_lip'], fill=(255, 0, 255, 255), width=5)
        
        

        # segment face
        plt.imshow(cv2.cvtColor(np.array(pil_frame), cv2.COLOR_BGR2RGB))
        # input points are both eyebrows as well as the center of the face (the nose) since I noticed the model has trouble with my glasses
        input_points = np.array([feature_to_point(face['left_eyebrow']), feature_to_point(face['right_eyebrow']), feature_to_point(face['nose_tip'])])

        # get mask
        predictor.set_image(np.array(pil_frame))
        masks, _, _ = predictor.predict(
            point_coords=input_points,
            point_labels=np.array([1, 1, 1]),
            multimask_output=False,
        )

        show_mask(masks[0], plt.gca())
        plt.title(f"Mask", fontsize=18)
        plt.axis('off')
    
    plt.pause(0.2)
    
#    # show frame
#    if cv2.waitKey(1) == ord('q'):
#        break
#    cv2.imshow('Segment Face', np.array(pil_frame))

#import cv2
#import matplotlib.pyplot as plt
#
#cap = cv2.VideoCapture(0)
#
#plt.ion()
#
#while cap.isOpened():
#    ret, frame = cap.read()
#    if not ret: break
#    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#    plt.pause(0.2)
