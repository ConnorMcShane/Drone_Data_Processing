import cv2
import numpy as np
from PIL import Image
import math


centre = (500, 300)
height = 200
width = 400
pt1 = (-int(width/2) + centre[0], -int(height/2) + centre[1])
pt2 = (int(width/2) + centre[0], int(height/2) + centre[1])

def draw_rectangle(image, centre, theta, width, height):
    theta = np.radians(theta)
    r = 1/(abs(math.sin(theta)*(width/height)) + abs(math.cos(theta)))#(1-height/width)
    width = width*r
    height = height*r
    c, s = np.cos(theta), np.sin(theta)
    R = np.matrix('{} {}; {} {}'.format(c, -s, s, c))
    # print(R)
    #print centre[0]
    p1 = [ + width / 2,  + height / 2]
    p2 = [- width / 2,  + height / 2]
    p3 = [ - width / 2, - height / 2]
    p4 = [ + width / 2,  - height / 2]
    p1_new = np.dot(p1, R)+ centre
    p2_new = np.dot(p2, R)+ centre
    p3_new = np.dot(p3, R)+ centre
    p4_new = np.dot(p4, R)+ centre
    #print p1_new
    img = cv2.line(image, (int(p1_new[0, 0]), int(p1_new[0, 1])), (int(p2_new[0, 0]), int(p2_new[0, 1])), (255, 0, 0), 1)
    img = cv2.line(img, (int(p2_new[0, 0]), int(p2_new[0, 1])), (int(p3_new[0, 0]), int(p3_new[0, 1])), (255, 0, 0), 1)
    img = cv2.line(img, (int(p3_new[0, 0]), int(p3_new[0, 1])), (int(p4_new[0, 0]), int(p4_new[0, 1])), (255, 0, 0), 1)
    img = cv2.line(img, (int(p4_new[0, 0]), int(p4_new[0, 1])), (int(p1_new[0, 0]), int(p1_new[0, 1])), (255, 0, 0), 1)
    #img = cv2.line(img, (int(p2_new[0, 0]), int(p2_new[0, 1])), (int(p4_new[0, 0]), int(p4_new[0, 1])), (255, 0, 0), 1)
    #img = cv2.line(img, (int(p1_new[0, 0]), int(p1_new[0, 1])), (int(p3_new[0, 0]), int(p3_new[0, 1])), (255, 0, 0), 1)

    return img

colour = (0, 255, 0)
thickness = 2

for i in range(360):
    theta = i - 90
    blank_img = np.zeros((600, 1000, 3))
    img = cv2.rectangle(blank_img, pt1, pt2, colour, thickness)
    img = draw_rectangle(img, centre, theta, width, height)

    cv2.imshow('blank', img)
    cv2.waitKey(50)

