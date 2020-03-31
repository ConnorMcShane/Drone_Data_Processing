import cv2
import numpy as np

cap = cv2.VideoCapture('C:/Users/cmcshan1/Documents/DroneFootage/Drone_Videos/A45/DJI_0059.MP4')
kernel = np.ones((5,5),np.uint8)

ret, frame = cap.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# frame = cv2.GaussianBlur(frame, (21, 21), 0)

for i in range(200):    
    ret, new_frame = cap.read()
    new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
#     new_frame = cv2.GaussianBlur(new_frame, (21, 21), 0)

    diff = cv2.absdiff(new_frame, frame)
    th_delta = cv2.threshold(diff, 60, 255, cv2.THRESH_BINARY)[1]
    th_delta = cv2.dilate(th_delta, None, iterations=0)
#     th_delta = cv2.morphologyEx(th_delta, cv2.MORPH_CLOSE, kernel)
    
#     _, cnts, _1 = cv2.findContours(th_delta.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             
#     for contour in cnts:
#         if cv2.contourArea(contour) < 100:
#             continue
#         (x, y, w, h) = cv2.boundingRect(contour)
#         cv2.rectangle(new_frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
    
    cv2.imshow('Im', th_delta)
    cv2.waitKey(1)
    
    frame = new_frame