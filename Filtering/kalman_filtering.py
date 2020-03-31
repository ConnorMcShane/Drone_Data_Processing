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
import csv
import matplotlib.pyplot as plt

def get_angle(delta_x, delta_y):

    if delta_x >= 0 and delta_y >= 0:
        if (delta_x) == 0:
            theta = 0
        else:
            theta = (90-math.degrees(math.atan((delta_y)/(delta_x))))
    elif delta_x >= 0 and delta_y < 0:
        if (delta_x) == 0:
            theta = (180)
        else:
            theta = (90+math.degrees(math.atan((-delta_y)/(delta_x))))
    elif delta_x < 0 and delta_y < 0:
        theta = (270-math.degrees(math.atan((-delta_y)/(-delta_x))))
    elif delta_x < 0 and delta_y >= 0:
        theta = (270+math.degrees(math.atan((delta_y)/(-delta_x))))
    
    return theta


def predict(x, y, x_v, y_v, x_a, y_a, fps):
    # Assume constant acceleration

    new_x_a = x_a
    new_y_a = y_a
    new_x_v = x_v + new_x_a*(1/fps)
    new_y_v = y_v + new_y_a*(1/fps)
    new_x = x + new_x_v*(1/fps)
    new_y = y + new_y_v*(1/fps)
    delta_x = new_x - x
    delta_y = new_y - y
    theta = get_angle(delta_x, delta_y)    

    return new_x, new_y, new_x_v, new_y_v, new_x_a, new_y_a, theta


def update(x, y, x_v, y_v, m_x, m_y, p_x, p_y, n_pos, fps):

    x_measurement_confidence = (n_pos/(n_pos+abs(m_x-p_x)))
    y_measurement_confidence = (n_pos/(n_pos+abs(m_y-p_y)))
    u_x = x_measurement_confidence*m_x + (1-x_measurement_confidence)*p_x
    u_y = y_measurement_confidence*m_y + (1-y_measurement_confidence)*p_y
    delta_x = u_x - x
    delta_y = u_y - y
    u_x_v = delta_x/(1/fps)
    u_y_v = delta_y/(1/fps)
    u_x_a = (u_x_v - x_v)/(1/fps)
    u_y_a = (u_y_v - y_v)/(1/fps)
    u_theta = get_angle(delta_x, delta_y)    

    return u_x, u_y, u_x_v, u_y_v, u_x_a, u_y_a, u_theta

tracked_file = r'C:/Users/mlpre/Desktop/DJI_0025_TRACKED_100.csv'
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

filtered_array = []
filtered_array.append([tracked_array[0][0][2], tracked_array[0][0][3], 0, 0, 0, 0, 0])
filtered_array.append([tracked_array[1][0][2], tracked_array[1][0][3], 0, 0, 0, 0, 0])
fps = 29
time_step = 1/fps
n_pos = 1000
for i in range(2, len(tracked_array)):
    _, _, x, y, w, h, w_avg, h_avg, t = tracked_array[i][0]
    x1, y1, x_v1, y_v1, x_a1, y_a1, t1 = filtered_array[i-1]
    x2, y2, x_v2, y_v2, x_a2, y_a2, t2 = filtered_array[i-2]
    x, y, w, h, w_avg, h_avg, t = int(float(x)), int(float(y)), int(float(w)), int(float(h)), int(float(w_avg)), int(float(h_avg)), int(float(t))
    x1, y1, x_v1, y_v1, x_a1, y_a1, t1 = int(float(x1)), int(float(y1)), int(float(x_v1)), int(float(y_v1)), int(float(x_a1)), int(float(y_a1)), int(float(t1))
    x2, y2, x_v2, y_v2, x_a2, y_a2, t2 = int(float(x2)), int(float(y2)), int(float(x_v2)), int(float(y_v2)), int(float(x_a2)), int(float(y_a2)), int(float(t2))
    
    x_v, y_v = (x-x1)/time_step, (y-y1)/time_step
    x_v1, y_v1 = (x1-x2)/time_step, (y1-y2)/time_step
    x_a, y_a = (x_v-x_v1)/time_step, (y_v-y_v1)/time_step
    new_x, new_y, new_x_v, new_y_v, new_x_a, new_y_a, theta = predict(x, y, x_v, y_v, x_a, y_a, fps)
    u_x, u_y, u_x_v, u_y_v, u_x_a, u_y_a, u_theta = update(x1, y1, x_v1, y_v1, x, y, new_x, new_y, n_pos, fps)
    filtered_array.append([round(u_x,2), round(u_y,2), round(u_x_v,2), round(u_y_v,2), round(u_x_a,2), round(u_y_a,2), round(u_theta,2)])

x_array = []
x_array_unfiltered = []
for i in range(len(filtered_array)):
    x_array.append(filtered_array[i][1])
    x_array_unfiltered.append(tracked_array[i][0][2])

plt.plot(x_array)
plt.ylabel('x')
plt.show()
print('done')