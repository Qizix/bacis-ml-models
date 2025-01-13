import numpy as np
import pandas as pd

# import OpenCV
import cv2

# Visualization
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter('ignore')

low_bound = np.array([10, 100, 20])
up_bound = np.array([25, 255, 255])


cap = cv2.VideoCapture(0)
if(cap.isOpened()):
    print("webCam opened")
    while(cv2.waitKey(3) != ord('q')):
        ret, frame = cap.read()
        # Convert color from BGR to HSV
        # Please see detail in https://en.wikipedia.org/wiki/HSL_and_HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)     
      
        # Filtering by color
        mask = cv2.inRange(hsv, low_bound, up_bound)
      
        # Image dilation
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations = 1)
      
        # Combined two images (filtering image and original image) in an one window
        h1, w1 = mask.shape
        h2, w2, d2 = frame.shape
        imgCom = np.zeros((max([h1, h2]), w1 + w2, 3), dtype='uint8')
        imgCom[:h1,:w1, 0] = mask
        imgCom[:h1,:w1, 1] = mask
        imgCom[:h1,:w1, 2] = mask
        imgCom[:h2, w1:w1+w2, :] = np.dstack([frame])
      
        # Image resize
        window_name = "webCam"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Create window with resize capability
        new_width = 1600
        new_height = 1200
        cv2.resizeWindow(window_name, new_width, new_height)
        cv2.imshow(window_name, imgCom)
else:
    print("Something is wrong")
  
cap.release()
cv2.destroyAllWindows()