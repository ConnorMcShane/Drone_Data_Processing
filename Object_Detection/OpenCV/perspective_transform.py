import time
import cv2
import numpy as np
from PIL import Image
import os
import math
from math import cos
from math import tan
from math import atan
from math import radians
from threading import Thread
from queue import Queue
import psutil
import threading

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

'''class FileVideoStream:
    def __init__(self, files, queueSize=500):
        self.files = files
        self.fileNumber = 0
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
                frame = cv2.imread(self.files[self.fileNumber])
                self.Q.put(frame)
                self.fileNumber = self.fileNumber + 1
                if self.fileNumber >= len(self.files):
                    self.stopped = True     

    def read(self):
	    return self.Q.get()

    def stop(self):
	    self.stopped = True'''

class ImageProcess:
    def __init__(self):
        self.stopped = False
        self.Q = Queue(maxsize=10)
        self.Q2 = Queue(maxsize=100)
        self.check = 0
        self.processed = 0

    def start(self):
        self.t = Thread(target=self.process, args=())
        self.t.daemon = True
        self.t.start()
        return self

    def send(self, img, cam, distCoeff, matrix, extendedWidth, extendedHeight):
        self.check = 0
        while self.check == 0:
            if not self.Q.full():
                self.Q.put([img, cam, distCoeff, matrix, extendedWidth, extendedHeight])
                self.check = 1
    
    def read(self):
        return self.Q2.get()

    def more(self):
	    return self.Q.qsize() > 0

    def process(self):
        while True:
            if self.stopped:
                return
            processInput = self.Q.get()
            img, cam, distCoeff, matrix, extendedWidth, extendedHeight = processInput[0], processInput[1], processInput[2], processInput[3], processInput[4], processInput[5]
            img = cv2.undistort(img,cam,distCoeff)
            img = cv2.warpPerspective(img, matrix, (extendedWidth, extendedHeight))
            self.Q2.put(img)
            #cv2.imwrite(path,img)
            self.processed = self.processed + 1
    
    def stop(self):
	    self.stopped = True

# Main function of the module
def perspective_transform(scr_video, dst_video, scale, numberOfFrames, fps, imgRatio, focalLength, barrel_correct, processing_threads):
    
    all_start_time = time.time()

    #for file in os.listdir(stabilizedFolder + r'\frames\\'):
    #    if file.endswith(r'.jpg'):
    #        files.append(os.path.join(stabilizedFolder + r'\frames\\', file))
    video_path = scr_video
    fvs = FileVideoStream(video_path, scale).start()
    img = fvs.read()
    #img = cv2.imread(files[0])
    h, w, _ = img.shape
    distCoeff = np.zeros((4,1),np.float64)
    k1, k2, p1, p2 = barrel_correct, 0.0, 0.0, 0.0
    distCoeff[0,0], distCoeff[1,0], distCoeff[2,0], distCoeff[3,0] = k1, k2, p1, p2
    cam = np.eye(3,dtype=np.float32)
    cam[0,2], cam[1,2], cam[0,0], cam[1,1] = w/2.0, h/2.0, focalLength, focalLength
    imgRatio = 1.2
    margin = int(w*0.05)
    widthInset = int(((w*imgRatio)-w)/2)
    extendedHeight = int(h*imgRatio)
    extendedWidth = int(w + margin*2)
    
    srcPts = np.float32([[widthInset, 0], [w-widthInset, 0], [0, h], [w, h]])
    dstPts = np.float32([[margin, 0], [w+margin, 0], [margin, extendedHeight], [w+margin, extendedHeight]])
    matrix = cv2.getPerspectiveTransform(srcPts, dstPts)

    #fvs = FileVideoStream(files).start()
    
    ImageProcessor = [None]*processing_threads
    for j in range(processing_threads):
        ImageProcessor[j] = ImageProcess().start()

    class VideoWrite:
        def __init__(self, width, height, fps, dst_video, processing_threads):
            self.stopped = False
            self.has_more = True
            self.released = False
            self.check = 0
            self.processed = 0
            self.video = cv2.VideoWriter(dst_video, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width,height))
            self.processing_threads = processing_threads

        def start(self):
            self.t = Thread(target=self.process, args=())
            self.t.daemon = True
            self.t.start()
            return self

        def process(self):
            while True:
                if self.stopped:
                    return
                for j in range(self.processing_threads):
                    self.check = 0
                    while self.check == 0 and self.processed < numberOfFrames:
                        if ImageProcessor[j].Q2.qsize() > 0:
                            img = ImageProcessor[j].Q2.get()
                            self.video.write(img)
                            self.processed = self.processed + 1
                            self.check = 1
                if self.processed >= numberOfFrames:
                    self.video.release()
                    time.sleep(1)
                    self.released = True
                        
        def stop(self):
            self.stopped = True

    video_writer = VideoWrite(extendedWidth, extendedHeight, fps, dst_video, processing_threads).start()
    for i in range(0, numberOfFrames+processing_threads, processing_threads):

        img = [None]*processing_threads
        for j in range(processing_threads):
            if i + j < numberOfFrames:
                try:
                    img[j] = fvs.read()
                except:
                    break

        img_number = [None]*processing_threads
        #path = [None]*processing_threads
        for j in range(processing_threads):
            if i + j < numberOfFrames:
                try:
                    img_number[j] = str(i+j)
                    for _ in range(5-len(img_number[j])):
                        img_number[j] = '0' + img_number[j]
                    #path[j] = warpedFolder + r'\frames\\' + r'Warped_image_' + img_number[j] + r'.jpg'
                    ImageProcessor[j].send(img[j], cam, distCoeff, matrix, extendedWidth, extendedHeight)
                except:
                    break
        
        percentage = video_writer.processed/numberOfFrames*100 #sum([ImageProcessor[x].processed for x in range(len(ImageProcessor))])/numberOfFrames*100
        cpu = psutil.cpu_percent()
        ram = dict(psutil.virtual_memory()._asdict())['percent']
        all_end_time = time.time()
        all_time = all_end_time - all_start_time
        loaded = '>'
        unloaded = '...................'
        for t in range(int(percentage/5)):
            unloaded = unloaded[:-1]
            loaded = '=' + loaded  
        print('\r' + 'Warping Images (' + str(processing_threads) + ' threads) [' + loaded + unloaded + ']  ' + str(int(percentage*100)/100) + '%  [CPU: ' + str(cpu) + '% | RAM: ' + str(ram) + '%' + ' | Total time: ' + str(round(all_time,2)) + 's]             ', end = '')
    
    while video_writer.released == False: #sum([ImageProcessor[x].processed for x in range(len(ImageProcessor))]) < numberOfFrames and sum([ImageProcessor[x].Q2.qsize() for x in range(len(ImageProcessor))]) > 0:
        percentage = video_writer.processed/numberOfFrames*100 #sum([ImageProcessor[x].processed for x in range(len(ImageProcessor))])/numberOfFrames*100
        cpu = psutil.cpu_percent()
        ram = dict(psutil.virtual_memory()._asdict())['percent']
        all_end_time = time.time()
        all_time = all_end_time - all_start_time
        loaded = '>'
        unloaded = '...................'
        for j in range(int(percentage/5)):
            unloaded = unloaded[:-1]
            loaded = '=' + loaded  
        print('\r' + 'Warping Images (' + str(processing_threads) + ' threads) [' + loaded + unloaded + ']  ' +  str(int(percentage*100)/100) + '%  [CPU: ' + str(cpu) + '% | RAM: ' + str(ram) + '%' + ' | Total time: ' + str(round(all_time,2)) + 's]             ', end = '')
    
    fvs.stop()
    for x in range(len(ImageProcessor)):
        ImageProcessor[x].stop()

    threads = []
    threads.append(fvs.t)
    for x in range(len(ImageProcessor)):
        threads.append(ImageProcessor[x].t)
    for t in threads:
        t.join(0)
    
    all_end_time = time.time()
    all_time = all_end_time - all_start_time
    print('\r' + 'Warping Images (' + str(processing_threads) + ' threads) [====================] 100.00% Total time: ' + str(round(all_time,2)) + 's                                                                         ')
