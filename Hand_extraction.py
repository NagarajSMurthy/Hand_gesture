# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 15:19:39 2020

@author: nagar
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

# global background variable
bg = None

# initialize weight for running average
aWeight = 0.8

# get the reference to the webcam
camera = cv2.VideoCapture(0)

# region of interest (ROI) coordinates
top, right, bottom, left = 30, 300, 300, 590

# initialize num of frames
num_frames = 0


def running_avg(image, beta):
    global bg
    
    # background initialization
    if bg is None:
        bg = image.copy().astype('float')
        return 
    
    cv2.accumulateWeighted(image, bg, beta)
    
    
def segment(image, threshold=25):
    
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    thresholded = cv2.erode(thresholded, None, iterations = 3)
    thresholded = cv2.dilate(thresholded, None, iterations = 4)
    # get the contours in the thresholded image
    cnts, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)
    
    
# keep looping, until interrupted
while(True):
    # get the current frame
    ret, frame = camera.read()

    # flip the frame so that it is not the mirror view
    frame = cv2.flip(frame, 1)
    #print(frame.shape)
    w,h,c = frame.shape
    # clone the frame
    clone = frame.copy()
    hand_only = 0   
    
    # get the height and width of the frame
    (height, width) = frame.shape[:2]

    # get the ROI
    roi = frame[top:bottom, right:left]
    #roi = cv2.resize(roi,(w,h))
    #print(roi.shape)
    roi_copy = roi.copy()
    
    # convert the roi to grayscale and blur it
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    #roi_copy = gray.copy()
    # to get the background, keep looking till a threshold is reached
    # so that our running average model gets calibrated
    if num_frames < 30:
        running_avg(gray, aWeight)
    else:
        # segment the hand region
        hand = segment(gray)

        # check whether hand region is segmented
        if hand is not None:
            # if yes, unpack the thresholded image and
            # segmented region
            (thresholded, segmented) = hand
            
            #print(thresholded.shape)
            #print(roi_copy.shape[1::-1])
            # draw the segmented region and display the frame
            hull = cv2.convexHull(segmented, returnPoints = False)
            defects = cv2.convexityDefects(segmented,hull)   
            
            try:
                
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                cv2.drawContours(clone, [hull + (right, top)], -1, (0,255,0))
                for i in range(defects.shape[0]):
                    s,e,f,d = defects[i,0]
                    
                    a,b = segmented[s][0]
                    start = (a+right,b+top)
                    
                    x,y = segmented[e][0]
                    end = (x+right,y+top)
                    
                    m,n = segmented[f][0]
                    far = (m+right,n+top)
                    
                    cv2.line(clone,start,end,[0,255,0],2)
                    cv2.circle(clone,far,5,[0,0,255],-1)
            
            except AttributeError as e:
                pass
        
            cv2.imshow("Thesholded", thresholded)
        
            thresh = [thresholded for i in range(3)]
            thresh = np.stack(thresh, axis = 2)
            #print(thresh.shape)
            dst = cv2.bitwise_and(roi_copy,thresh)
            
            cv2.imshow('Hand only',dst)
            
            
    # draw the segmented hand
    cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

    # increment the number of frames
    num_frames += 1

    # display the frame with segmented hand
    cv2.imshow("Video Feed", clone)

    # observe the keypress by the user
    keypress = cv2.waitKey(1) & 0xFF

    # if the user pressed "q", then stop looping
    if keypress == ord("q"):
        break

# free up memory
camera.release()
cv2.destroyAllWindows()
