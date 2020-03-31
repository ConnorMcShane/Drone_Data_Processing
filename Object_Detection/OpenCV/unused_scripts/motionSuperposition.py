import cv2
import numpy as np
from PIL import Image
import math
from scipy.spatial import distance as dist
import sys
import csv
import pickle
import os

# Main function of the module
def motionSuperposition(rootFolder, videoName, segmentLengthSeconds, numberOfFrames):
    
    stabilizedFolder = rootFolder + r'\stabilized\\' + videoName + r'\\'
    backgroundFolder = rootFolder + r'\background\\' + videoName + r'\\'
    detectionsFolder = rootFolder + r'\detections\\' + videoName + r'\\'
    originalFolder = rootFolder + r'\original\\'
    warpedFolder = rootFolder + r'\warped\\' + videoName + r'\\'
    trajectoriesFolder = rootFolder +  r'\trajectories\\' + videoName + r'\\'

    folders = [rootFolder, rootFolder + r'\background\\', rootFolder + r'\stabilized\\', stabilizedFolder, rootFolder + r'\detections\\', rootFolder +  r'\trajectories\\', backgroundFolder, detectionsFolder, detectionsFolder + r'\frames\\', detectionsFolder + r'\videos\\', originalFolder, trajectoriesFolder,  rootFolder + r'\warped\\', warpedFolder, warpedFolder + r'\frames\\' , warpedFolder + r'\videos\\']

    video = warpedFolder + r'\videos\\' + videoName + r'.avi'

    for folder in folders:
        if not os.path.isdir(folder):
            os.mkdir(folder)
    
    cap = cv2.VideoCapture(video)
    kernel = np.ones((3,3),np.uint8)
    kernelErode = np.ones((2,2),np.uint8)
    kernelDilate = np.ones((9,9),np.uint8)
    kernelClose = np.ones((5,5),np.uint8)
    kernelOpen = np.ones((5,5),np.uint8)
    kernelCloseFinal = np.ones((3,3),np.uint8)
   
    # Initial variables
    segmentation_seconds = segmentLengthSeconds
    frame_rate = 29
    frameStep = 1
    frame_index = 0
    vehicles_previous = []
    new_centroids = np.ones((0,0))
    tracked_vehicles = []
    tracked_vehicle_counter = 0
    number_of_speeds = int(frame_rate*1)
    pixel_to_meter = 50/7.5
    mps_mph = 2.237
    laneWidth = 53
    frame_counter = 0 
    superposeQty = 7


    # Loop through each seperate background segment
    for j in range(int(numberOfFrames/(frame_rate*segmentation_seconds))):
        # Load in background for segment
        background = cv2.imread(backgroundFolder + r'\\' + videoName + r'_background_' + str(j) + r'.png', cv2.IMREAD_GRAYSCALE)
        print('Using background: ' + str(j))
        # Loop through each frame in background segment
        for i in range((j*frame_rate*segmentation_seconds),((j+1)*(frame_rate*segmentation_seconds)),frameStep):
            cap.set(1, i)
            ret, new_frame = cap.read()
            new_frame_clr = new_frame
            new_frame_clr2 = new_frame
            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(new_frame, background) 
            imgs = [diff,diff,diff,diff,diff,diff]

            for t in range(6):
                # Load in fram and subtract the background
                cap.set(1, i+t)
                ret, new_frame = cap.read()
                new_frame_clr = new_frame
                new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(new_frame, background)        

                # Morphology operations
                th_delta = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)[1]
                th_delta = cv2.GaussianBlur(th_delta,(33,33),cv2.BORDER_DEFAULT)
                #th_delta = cv2.dilate(th_delta,kernel,iterations = 3)          
                th_delta = cv2.threshold(th_delta, 125, 255, cv2.THRESH_BINARY)[1]
                
                imgs[t] = th_delta
                img = ((imgs[0]/6)+(imgs[1]/6)+(imgs[2]/6)+(imgs[3]/6)+(imgs[4]/6)+(imgs[5]/6))
                img = img.astype(dtype="uint8")
            
            #img = cv2.dilate(img,kernel,iterations = 3)  
            img2 = cv2.GaussianBlur(img,(33,33),cv2.BORDER_DEFAULT)
            #img2 = cv2.dilate(img2,kernel,iterations = 3)
            img2 = cv2.threshold(img2, 150, 255, cv2.THRESH_BINARY)[1]
            
            
            a, contours, hierarchy = cv2.findContours(th_delta, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                (x, y, w, h) = cv2.boundingRect(contour)
                # Set object tracking boundry to only include the road and ignore the very edge of the image
                if new_frame.shape[0]*0.05 < y < new_frame.shape[0]*0.95 and new_frame.shape[1]*0.05 < x and x + w < new_frame.shape[1]*0.95:
                    if  300 < cv2.contourArea(contour) < 15000:
                        cv2.rectangle(new_frame_clr , (x, y), (x+w, y+h), (255, 255, 255), 2)
            cv2.imshow('test', new_frame_clr)
            cv2.waitKey(1)


rootFolder = r'C:\Users\cmcshan1\Documents\DroneFootage\Drone_Videos'
videoName = r'DJI_0059'
segmentLengthSeconds = 20
numberOfFrames = 6960
fps = 29
motionSuperposition(rootFolder, videoName, segmentLengthSeconds, numberOfFrames)