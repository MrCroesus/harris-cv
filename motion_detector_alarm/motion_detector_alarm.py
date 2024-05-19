import cv2
import pygame
import os
import datetime

# initialize variables
pygame.init()
pygame.mixer.init()
sound = pygame.mixer.Sound("alert.wav")

MOVEMENT_THRESHOLD = 20
CONTOUR_THRESHOLD = 5000


# webcam input
cv2.namedWindow("Motion Detector Alarm")
cap = cv2.VideoCapture(0)


# video parameters
fps = int(cap.get(cv2.CAP_PROP_FPS))
screenshot_taken_count = 0


while cap.isOpened():
    # get frames
    ret1, frame1 = cap.read()
    if not ret1: break

    ret2, frame2 = cap.read()
    if not ret2: break

    # compute motion
    diff = cv2.absdiff(frame1, frame2)

    # smooth and threshold
    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, MOVEMENT_THRESHOLD, 255, cv2.THRESH_BINARY)

    # find and draw contours
    im = frame1.copy()
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(im, contours, -1, (255, 255, 255), 3)

    contour_areas = list(map(cv2.contourArea, contours))
    if len(contour_areas) > 0 and max(contour_areas) > CONTOUR_THRESHOLD:
        # play sound if over threshold
        sound.play()

        # send image if over threshold
        if screenshot_taken_count == 0:
            timestamp = '{:%Y-%m-%d %H,%M,%S}'.format(datetime.datetime.now())
            fname = 'out_path/' + timestamp + '.png'
            cv2.imwrite(fname, frame1)
            os.system('osascript sendFile.scpt 6269919887 "Motion Detected" ' + '"' + fname + '"')
            screenshot_taken_count += 1
        else:
            screenshot_taken_count = (screenshot_taken_count + 1) % fps


    # show frame
    if cv2.waitKey(1) == ord('q'):
        break
    cv2.imshow('Motion Detector Alarm', im)
