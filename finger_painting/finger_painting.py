import numpy as np
import cv2
import mediapipe as mp
import os
import matplotlib.pyplot as plt
from enum import Enum
import math
import datetime
import time



# initialize variables
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

width = 1280
height = 720
fps = 25



# webcam input
cv2.namedWindow("Finger Painting")
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FPS, fps)



# image
temp_canvas = np.zeros((height, width, 3), np.uint8)
canvas = np.zeros((height, width, 3), np.uint8)

# get header overlays
folder_path = 'header'
header_list = []

for im_path in os.listdir(folder_path):
    header = cv2.imread(folder_path + '/' + im_path)
    header_list.append(header)



# options
header_im = header_list[0]
color = (0, 0, 255)
thickness = 20
fingertip_ids = [4, 8, 12, 16, 20]
x_prev, y_prev = [0, 0]

# helper variables
color_x = 0
thickness_selected = False
fingers_up = [False, False, False, False, False]
screenshot_taken_count = 0



def get_landmarks_and_fingers_up(hand_landmarks):
    # get landmarks from hand landmarks
    landmarks = []
    for landmark in hand_landmarks.landmark:
        landmarks.append([int(landmark.x * width), int(landmark.y * height)])

    fingers_up = []

    # check if thumb is up
    right_hand = landmarks[fingertip_ids[1]][0] < landmarks[fingertip_ids[4]][0]
    left_hand = landmarks[fingertip_ids[1]][0] > landmarks[fingertip_ids[4]][0]
    right_thumb_out = landmarks[fingertip_ids[0]][0] < landmarks[fingertip_ids[0] - 1][0]
    left_thumb_out = landmarks[fingertip_ids[0]][0] > landmarks[fingertip_ids[0] - 1][0]

    if (left_hand and left_thumb_out) or (right_hand and right_thumb_out):
        fingers_up.append(True)
    else:
        fingers_up.append(False)

    # check if fingers are up
    for id in range(1, 5):
        if landmarks[fingertip_ids[id]][1] < landmarks[fingertip_ids[id] - 2][1]:
            fingers_up.append(True)
        else:
            fingers_up.append(False)
            
    return landmarks, fingers_up



Mode = Enum('Mode', ['Draw', 'Resize', 'Select', 'Standby', 'Clear'])



def get_mode(fingers_up):
    draw = [False, True, False, False, False]
    resize = [True, True, False, False, False]
    select = [False, True, True, False, False]
    standby = [False, True, False, False, True]
    clear = [False, False, False, False, False]
    
    if all([fingers_up[i] == draw[i] for i in range(5)]):
        return Mode.Draw
    elif all([fingers_up[i] == resize[i] for i in range(4)]):
        return Mode.Resize
    elif all([fingers_up[i] == select[i] for i in range(5)]):
        return Mode.Select
    elif all([fingers_up[i] == standby[i] for i in range(5)]):
        return Mode.Standby
    elif all([fingers_up[i] == clear[i] for i in range(5)]):
        return Mode.Clear
    


def constrain(val, min_val, max_val):
    return min(max(val, min_val), max_val)



with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1) as hands:
    while cap.isOpened():
        # get frame
        ret, frame = cap.read()
        if not ret: break

        # flip frame for mirrored view
        frame = cv2.flip(frame, 1)
        
        # change to rgb and get hands
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            # get hand landmarks from multi hand landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks, fingers_up = get_landmarks_and_fingers_up(hand_landmarks)

                mode = get_mode(fingers_up)

                match mode:
                    case Mode.Draw:
                        # draw circle on index finger to indicate draw mode
                        cv2.circle(temp_canvas, landmarks[fingertip_ids[1]], thickness // 2, color, cv2.FILLED)

                        # update previous position if it doesn't exist
                        if x_prev == 0 and y_prev == 0:
                            x_prev, y_prev = landmarks[fingertip_ids[1]]

                        # draw line, with eraser being thicker
                        if color == (0, 0, 0):
                            cv2.line(canvas, (x_prev, y_prev), landmarks[fingertip_ids[1]], color, thickness * 3)
                        else:
                            cv2.line(canvas, (x_prev, y_prev), landmarks[fingertip_ids[1]], color, thickness)

                        # update previous position
                        x_prev, y_prev = landmarks[fingertip_ids[1]]

                    case Mode.Resize:
                        x0, y0 = landmarks[fingertip_ids[0]]
                        x1, y1 = landmarks[fingertip_ids[1]]

                        radius = int(math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2) / 3)

                        if fingers_up[4] and not thickness_selected:
                            thickness = radius
                            thickness_selected = True
                        elif fingers_up[4] and thickness_selected:
                            cv2.circle(temp_canvas, [(x0 + x1) // 2, (y0 + y1) // 2], thickness // 2, color, cv2.FILLED)
                            cv2.putText(temp_canvas, "Thickness Selected!", landmarks[fingertip_ids[4]], cv2.FONT_HERSHEY_SIMPLEX, 1, (255 - color[0], 255 - color[1], 255 - color[2]))
                        elif not fingers_up[4]:
                            cv2.circle(temp_canvas, [(x0 + x1) // 2, (y0 + y1) // 2], radius // 2, color, cv2.FILLED)
                            thickness_selected = False

                    case Mode.Select:
                        x = (landmarks[fingertip_ids[1]][0] + landmarks[fingertip_ids[2]][0]) // 2
                        y = (landmarks[fingertip_ids[1]][1] + landmarks[fingertip_ids[2]][1]) // 2

                        if y < 125:
                            if 980 < x < 1105:
                                header_im = header_list[1]
                                color = (0, 0, 0)
                            elif x < 980:
                                # calculate red
                                red = 315 - abs(x - 232.5)
                                red = constrain(red, 0, 255)

                                # calculate green
                                green = 315 - abs(x - 497.5)
                                green = constrain(green, 0, 255)

                                # calculate blue
                                blue = 315 - abs(x - 762.5)
                                blue = constrain(blue, 0, 255)

                                # set color, color_x, and header
                                color = (blue, green, red)
                                color_x = x

                                header_im = header_list[0]
                            
                        cv2.rectangle(temp_canvas, landmarks[fingertip_ids[1]], landmarks[fingertip_ids[2]], color, cv2.FILLED)

                    case Mode.Standby:
                        cv2.line(temp_canvas, (x_prev, y_prev), landmarks[fingertip_ids[4]], color, 5)
                        x_prev, y_prev = landmarks[fingertip_ids[1]]

                    case Mode.Clear:
                        canvas = np.zeros_like(canvas)
                        x_prev, y_prev = landmarks[fingertip_ids[1]]

        # set header and highlight selection on header
        frame[:125, :width] = header_im
        cv2.rectangle(temp_canvas, (color_x - 5, -5), (color_x + 5, 130), (255, 255, 255), 5)

        # generate temp mask
        temp_canvas_bw = cv2.cvtColor(temp_canvas, cv2.COLOR_BGR2GRAY)
        _, inv_temp_mask_bw = cv2.threshold(temp_canvas_bw, 5, 255, cv2.THRESH_BINARY_INV)
        inv_temp_mask = cv2.cvtColor(inv_temp_mask_bw, cv2.COLOR_GRAY2BGR)

        # apply temp canvas to frame and clear temp canvas
        im = cv2.bitwise_and(frame, inv_temp_mask)
        im = cv2.bitwise_or(im, temp_canvas)
        temp_canvas = np.zeros_like(temp_canvas)

        # generate mask
        canvas_bw = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, inv_mask_bw = cv2.threshold(canvas_bw, 5, 255, cv2.THRESH_BINARY_INV)
        inv_mask = cv2.cvtColor(inv_mask_bw, cv2.COLOR_GRAY2BGR)

        # apply canvas to frame
        im = cv2.bitwise_and(im, inv_mask)
        im = cv2.bitwise_or(im, canvas)

        # take screenshot
        if results.multi_hand_landmarks and all(fingers_up) and screenshot_taken_count == 0:
            fname = 'out_path/screenshot' + str(datetime.datetime.now()) + '.png'
            cv2.imwrite(fname, im)
            screenshot_taken_count += 1
        else:
            screenshot_taken_count = (screenshot_taken_count + 1) % fps

        # show frame
        cv2.imshow("Finger Painting", im)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
