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

    def read(self):
        return self.Q.get()

    def more(self):
	    return self.Q.qsize() > 0
    
    def stop(self):
	    self.stopped = True

root = Tk()
root.withdraw()
file_path = filedialog.askdirectory(title = 'Select Video Folder', initialdir = str(pathlib.Path(__file__).parent.absolute()))
file_names = os.listdir(file_path)
files = [os.path.join(file_path, f) for f in file_names]
downsampled_files = [os.path.join(file_path, f[:-4] + '_DOWNSAMPLED.AVI') for f in file_names]

scale = 1/2
for i in range(len(files)):
    video_path = files[i]
    fvs = FileVideoStream(video_path, scale).start()

    img = fvs.read()
    height, width, _ = img.shape
    dstVideo = downsampled_files[i]
    fps = 30
    video = cv2.VideoWriter(dstVideo, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width,height))
    j = 0
    while not fvs.stopped == True:
        j = j + 1
        print('Video number: ' + str(i) + '. Frame number: ' + str(j) + '            ', end = '\r')
        video.write(fvs.read())
video.release()
print('done')


