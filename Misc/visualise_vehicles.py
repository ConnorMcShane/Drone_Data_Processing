import cv2
import numpy as np
from PIL import Image
import math
from scipy.spatial import distance as dist
import sys
import csv
import pickle
import os
from threading import Thread
from queue import Queue
import psutil
import threading
import time
from tkinter import filedialog
from tkinter import Tk
import pathlib

def draw_rectangle(image, centre, theta, width, height, colour = (0,255,0), thickness = 2):
    theta = np.radians(theta)

    c, s = np.cos(theta), np.sin(theta)
    R = np.matrix('{} {}; {} {}'.format(c, -s, s, c))

    p1 = [ + width / 2,  + height / 2]
    p2 = [- width / 2,  + height / 2]
    p3 = [ - width / 2, - height / 2]
    p4 = [ + width / 2,  - height / 2]
    p1_new = np.dot(p1, R)+ centre
    p2_new = np.dot(p2, R)+ centre
    p3_new = np.dot(p3, R)+ centre
    p4_new = np.dot(p4, R)+ centre

    img = cv2.line(image, (int(p1_new[0, 0]), int(p1_new[0, 1])), (int(p2_new[0, 0]), int(p2_new[0, 1])), colour, thickness)
    img = cv2.line(img, (int(p2_new[0, 0]), int(p2_new[0, 1])), (int(p3_new[0, 0]), int(p3_new[0, 1])), colour, thickness)
    img = cv2.line(img, (int(p3_new[0, 0]), int(p3_new[0, 1])), (int(p4_new[0, 0]), int(p4_new[0, 1])), colour, thickness)
    img = cv2.line(img, (int(p4_new[0, 0]), int(p4_new[0, 1])), (int(p1_new[0, 0]), int(p1_new[0, 1])), colour, thickness)

    return img


def draw_2d(img, x, y, w_avg, h_avg, text = '', size = 0.4, colour = (0,255,0), thickness = 2):
    pt1 = (int(x - w_avg/2), int(y - h_avg/2))
    pt2 = (int(x + w_avg/2), int(y + h_avg/2))
    img = cv2.rectangle(img, pt1, pt2, colour, thickness)

    if len(text) > 0:
        pt1 = (int(x - w_avg/2), int((y - h_avg/2)-(24*size)))
        pt2 = (int((x - w_avg/2)+(len(text)*24*size)), int(y - h_avg/2))
        img = cv2.rectangle(img , pt1, pt2, colour, -1)
        img = cv2.putText(img, text, (int(x - w_avg/2), int(y - h_avg/2 -1)), 0, size, (0, 0, 0), 1, cv2.LINE_AA)
    return img

def draw_3d(img, x, y, w_avg, h_avg, text, size = 0.4, colour = (0,255,0), thickness = 2, thickness2 = 1):
    img_h, img_w, _ = img.shape
    x_ratio = ((img_w/2 - x)/(img_w/2))/6
    y_ratio = ((img_h - y)/img_h)/4 + 0.15
    perspective_ratio = ((y+img_h*2)/(y+h_avg+img_h*2))/40
    bottom_ratio = (1-h_avg/(img_h*2))-0.025

    x, y = int(x-w_avg/2), int(y-h_avg/2)
    if x_ratio > 0:
        top_w3d = int(w_avg-(x_ratio*h_avg))
        top_x3d = x
        bottom_w3d = int((top_w3d*bottom_ratio))
        bottom_x3d =  int(x + (w_avg-top_w3d) + ((top_w3d-bottom_w3d)/2))
    else:
        top_w3d = int(w_avg+(x_ratio*h_avg))
        top_x3d = x + (w_avg-top_w3d)
        bottom_w3d = int((top_w3d*bottom_ratio))
        bottom_x3d = int(x + ((top_w3d-bottom_w3d)/2))

    top_y3d = y
    top_h3d = int(h_avg*(1-y_ratio))
    bottom_h3d = top_h3d
    bottom_y3d = y + (h_avg-top_h3d)

    top_back_x3d = int(top_x3d+(((perspective_ratio)/2)*top_h3d))
    bottom_back_x3d = int(bottom_x3d+(((perspective_ratio)/2)*bottom_h3d))
    top_back_w3d = int(top_w3d-(top_h3d*(perspective_ratio)))
    bottom_back_w3d = int(bottom_w3d-(bottom_h3d*(perspective_ratio)))

    img = cv2.line(img, (top_x3d, top_y3d + top_h3d), (bottom_x3d, bottom_y3d+bottom_h3d), colour, thickness) #left front verticle line
    img = cv2.line(img, (bottom_x3d, bottom_y3d+bottom_h3d), (bottom_x3d + bottom_w3d, bottom_y3d+bottom_h3d), colour, thickness) #bottom front horizontal line
    img = cv2.line(img, (bottom_x3d + bottom_w3d, bottom_y3d+bottom_h3d), (top_x3d + top_w3d, top_y3d+top_h3d), colour, thickness) #right front verticle line
    img = cv2.line(img, (top_x3d, top_y3d + top_h3d), (top_x3d + top_w3d, top_y3d + top_h3d), colour, thickness) #top front horizontal line
    img = cv2.line(img, (top_back_x3d + top_back_w3d, top_y3d), (top_x3d + top_w3d, top_y3d + top_h3d), colour, thickness) #right top horizontal line
    img = cv2.line(img, (top_back_x3d, top_y3d), (top_x3d, top_y3d + top_h3d), colour, thickness) #left top horizontal line
    img = cv2.line(img, (bottom_back_x3d, bottom_y3d), (bottom_back_x3d + bottom_back_w3d, bottom_y3d), colour, thickness2) #bottom rear horizontal line
    img = cv2.line(img, (top_back_x3d, top_y3d), (top_back_x3d + top_back_w3d, top_y3d), colour, thickness) #top rear horizontal line

    if bottom_back_x3d + bottom_back_w3d < top_x3d + top_w3d:
        img = cv2.line(img, ((top_back_x3d + top_back_w3d, top_y3d)), (bottom_back_x3d + bottom_back_w3d, bottom_y3d), colour, thickness2) #right rear verticle line
        img = cv2.line(img, ((bottom_back_x3d + bottom_back_w3d, bottom_y3d)), (bottom_x3d + bottom_w3d, bottom_y3d+bottom_h3d), colour, thickness2) #right bottom horizontal line
    else:
        img = cv2.line(img, ((top_back_x3d + top_back_w3d, top_y3d)), (bottom_back_x3d + bottom_back_w3d, bottom_y3d), colour, thickness) #right rear verticle line
        img = cv2.line(img, ((bottom_back_x3d + bottom_back_w3d, bottom_y3d)), (bottom_x3d + bottom_w3d, bottom_y3d+bottom_h3d), colour, thickness) #right bottom horizontal line
    if bottom_back_x3d > top_x3d:                
        img = cv2.line(img, ((bottom_back_x3d, bottom_y3d)), (top_back_x3d, top_y3d), colour, thickness2) #left rear verticle line
        img = cv2.line(img, ((bottom_back_x3d, bottom_y3d)), (bottom_x3d, bottom_y3d+bottom_h3d), colour, thickness2) #left bottom horizontal line
    else:
        img = cv2.line(img, ((bottom_back_x3d, bottom_y3d)), (top_back_x3d, top_y3d), colour, thickness) #left rear verticle line
        img = cv2.line(img, ((bottom_back_x3d, bottom_y3d)), (bottom_x3d, bottom_y3d+bottom_h3d), colour, thickness) #left bottom horizontal line

    pt1 = (int(x), int((y)-(24*size)))
    pt2 = (int((x)+(len(text)*24*size)), int(y))
    img = cv2.rectangle(img , pt1, pt2, colour, -1)
    img = cv2.putText(img, text, (int(x), int(y -1)), 0, size, (0, 75, 0), 1, cv2.LINE_AA)
    return img



def draw_trails(img, x, y, trails, width = 0.5, radius = 3, trail_frames = 30, delay = 30, thickness = 2):
    trails.append([x, y, trail_frames + delay])
    for t in range(len(trails)-1, 0, - 1):
        trails[t][2] = trails[t][2] - 1
        if trails[t][2] == 0:
            del trails[t]
        elif trails[t][2] < trail_frames:
            c = int(255*(trails[t][2]/trail_frames))
            colour = (0,127 + int(c/2),255-c)
            img = cv2.rectangle(img , (trails[t][0], int(trails[t][1] - width)), (trails[t][0] + 10, int(trails[t][1] + width)), colour, -1)
            img = cv2.circle(img , (trails[t][0], int(trails[t][1])), radius, colour, -1)
    return img, trails

# Main function of the module
def visualise(fps = 30, wait_key = 20, draw2d = True, draw3d = False, drawTrails = True, draw_angles = False, make_video = False):

    root = Tk()
    root.withdraw()
    current_file = '/'.join(str(pathlib.Path(__file__).parent.absolute()).split('\\')[:-3])
    video_file = filedialog.askopenfilename(title = 'Select Video', initialdir = current_file)
    tracked_file = filedialog.askopenfilename(title = 'Select CSV file', initialdir = current_file)
    video_name = video_file.split('/')[-1]
    video_folder = '/'.join(video_file.split('/')[:-1])
    dst_video = video_folder + '/' + 'visualised_' + video_name

    # Initial variables
    stream = cv2.VideoCapture(video_file)
    tracked_array = []
    temp_tracked_array = []
    previous_timestamp = None
    with open(tracked_file, 'r', newline='') as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            timestamp = row[0]
            if timestamp == previous_timestamp:
                temp_tracked_array.append(row)
            else:
                tracked_array.append(temp_tracked_array)
                temp_tracked_array = []
                temp_tracked_array.append(row)
            previous_timestamp = timestamp
    tracked_array = tracked_array[2:]
    number_of_frames = len(tracked_array)
    print('csv read in')

    trails = []
    for i in range(number_of_frames):
        (_, img) = stream.read()
        img_h, img_w, _ = img.shape
        if i == 0 and make_video == True:
            video = cv2.VideoWriter(dst_video, cv2.VideoWriter_fourcc(*'DIVX'), fps, (img_w,img_h))
        
        for v in tracked_array[i]:
            if len(v) == 9:
                _, vid, x, y, w, h, w_avg, h_avg, t = v
            else:
                _, vid, x, y, w, h = v
                w_avg, h_avg, t = w, h, 0
            if float(vid) >= 0:
                text = str(vid)
                x, y, w, h, w_avg, h_avg, t = int(float(x)), int(float(y)), int(float(w)), int(float(h)), int(float(w_avg)), int(float(h_avg)), int(float(t))

                if draw2d == True:
                    img = draw_2d(img, x, y, w_avg, h_avg, text)
                
                if draw3d == True:
                    img = draw_3d(img, x, y, w_avg, h_avg, text)
                
                if draw_angles == True:
                    img = draw_rectangle(img, (x, y), t-90, w_avg, h_avg)

                if drawTrails == True:
                    img, trails = draw_trails(img, x, y, trails, trail_frames=200, radius=2, width=2, delay =0)
        if make_video == True:
            video.write(img)
        cv2.imshow('tracked', img)
        cv2.waitKey(wait_key)
    if make_video == True:
        video.release()
        
visualise(make_video = True)