from darknet.darknet import performDetect
import sys
sys.path.insert(1, 'D:/Connor/Autoplex/Repos/Drone_Data_Processing/')
import random
import cv2
import time
import numpy as np
import os
from threading import Thread
from queue import Queue
import csv
from tkinter import filedialog
from tkinter import Tk
import pathlib
from Misc import visualise_vehicles


class FileVideoStream:
    def __init__(self, path, scale, queueSize=500):
        self.stream = cv2.VideoCapture(path)
        self.scale = scale
        self.stopped = False
        self.Q = Queue(maxsize=queueSize)

    def start(self):
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True
        self.t.start()
        return self
    
    def update(self):
        while True:
            if self.stopped:
                return
            if not self.Q.full():
                (grabbed, frame) = self.stream.read()
                if not grabbed:
                    self.stop()
                    return
                self.dims = int(frame.shape[1]*self.scale), int(frame.shape[0]*self.scale)
                frame = cv2.resize(frame, self.dims)
                self.Q.put(frame)
            else:
                time.sleep(5)

    def read(self, temp_img_path):
        self.img = self.Q.get()
        cv2.imwrite(temp_img_path, self.img)
        return  self.img

    def more(self):
	    return self.Q.qsize() > 0
    
    def stop(self):
	    self.stopped = True

# GET FILE PATHS
root = Tk()
root.withdraw()
darknet_root = filedialog.askdirectory(title = 'Select Darknet Folder', initialdir = str(pathlib.Path(__file__).parent.absolute()))
paths = '/'.join(str(darknet_root).split('/')[:-5]) 
video_path = filedialog.askopenfilename(title = 'Select Video', initialdir = '/'.join(str(darknet_root).split('/')[:-6]))
detections_folder = filedialog.askdirectory(title = 'Select Detections Folder', initialdir = '/'.join(str(darknet_root).split('/')[:-6]))
video_name = video_path.split('/')[-1][:-4]
video_folder = video_path[0:-(len(video_name)+4)]
dstVideo = detections_folder + r'/YOLOV3_' + video_name + r'.avi'
dst_csv = detections_folder + r'/' + video_name + r'.csv'

# CONFIG
scale = 1
fps = 30
make_video = True
show_images = True

# INITIALISE VIDEO STREAM ANDDETECTION ARRAY
fvs = FileVideoStream(video_path, scale).start()
detections_array = []
detections_array.append(['time_step (s)', 'vehicle_id', 'centre_x', 'centre_y', 'width', 'height'])
i = 0

while fvs.stopped == False:
    
    temp_img_path = darknet_root + r'/data/autoplex/temp/temp.jpg'
    img = fvs.read(temp_img_path)
    img_h, img_w, img_c =  img.shape
    if i == 0 and make_video == True:
        video = cv2.VideoWriter(dstVideo, cv2.VideoWriter_fourcc(*'DIVX'), fps, (img_w,img_h))

    detections = performDetect(temp_img_path, thresh = 0.2, configPath= darknet_root +"/cfg/yolov3-autoplex-test.cfg", weightPath= darknet_root + "/backup/autoplex/yolov3-autoplex_300000.weights", metaPath= darknet_root + "/data/autoplex/autoplex.data", showImage=False, makeImageOnly=False, initOnly=False)
    
    for j in range(len(detections)):
        dims = detections[j][2]
        x, y, w, h = int(dims[0]), int(dims[1]), int(dims[2]), int(dims[3])
        detections_array.append([str(round(i/fps,3)), str(j), str(x), str(y), str(w), str(h)])  
        img = visualise_vehicles.draw_2d(img, x, y, w, h)

    if make_video == True:
        video.write(img)
    if show_images == True:
        cv2.imshow('detections', img)
        cv2.waitKey(1)
    print('Detecting frame: ' + str(i) + '              ', end = '\r')
    i = i + 1

# RELEASE VIDEO AND SAVE DETECTION CSV
if make_video == True:
    video.release()
with open(dst_csv, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(detections_array)