import time
import cv2
import numpy as np
import os
from threading import Thread
from queue import Queue
import psutil
import threading
import math

maxFeatures = 0
maxPointsPixles = 5.0 #reduce for better find homography results 
contrastThresh = 0.02
edgeThresh = 2
sigmaValue = 1.2
nextBestMatchDistanceThresh = 0.7 # match dist must be less than next best match dist * nextBestMatchDistanceThresh
minHessian = 400

class FileVideoStream:
    def __init__(self, path, queueSize=500):
        self.stream = cv2.VideoCapture(path)
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
                self.Q.put(frame)

    def read(self):
	    return self.Q.get()

    def more(self):
	    return self.Q.qsize() > 0
    
    def stop(self):
	    self.stopped = True

class ImageProcess:
    def __init__(self):
        self.stopped = False
        self.Q = Queue(maxsize=400)
        self.check = 0
        self.processed = 0
        self.matched = 0

    def start(self):
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True
        self.t.start()
        return self
    
    def process(self, firstFrame_rgb_hires, frame_rgb_hires, frame, desc, kp, Mabs, siftDownsample, path):
        self.check = 0
        while self.check == 0:
            if not self.Q.full():
                self.Q.put([firstFrame_rgb_hires, frame_rgb_hires, frame, desc, kp, Mabs, siftDownsample, path])
                self.check = 1

    def more(self):
	    return self.Q.qsize() > 0

    def update(self):
        while self.stopped == False:
            if self.stopped:
                return
            sift = cv2.xfeatures2d.SIFT_create(nfeatures = maxFeatures, nOctaveLayers = 3, contrastThreshold = contrastThresh, edgeThreshold = edgeThresh, sigma = sigmaValue)
            #surf = cv2.xfeatures2d_SURF.create(hessianThreshold=minHessian)
            processInput = self.Q.get()
            firstFrame_rgb_hires, frame_rgb_hires, frame, desc, kp, Mabs, siftDownsample, path = processInput[0], processInput[1], processInput[2], processInput[3], processInput[4], processInput[5], processInput[6] , processInput[7]
            new_kp, new_desc = sift.detectAndCompute(frame,None)
            #new_kp_surf, new_desc_surf = surf.detectAndCompute(frame, None)
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks = 50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(desc,new_desc,k=2)
            good = []
            for m,n in matches:
                if m.distance < nextBestMatchDistanceThresh*n.distance:
                    good.append(m)
            
            '''# ratio test as per Lowe's paper
            matchesMask = [[0,0] for i in range(len(matches))]
            for i,(m,n) in enumerate(matches):
                if m.distance < nextBestMatchDistanceThresh*n.distance:
                    matchesMask[i]=[1,0]
            
            draw_params = dict(matchColor = (0,255,0), singlePointColor = (255,0,0), matchesMask = matchesMask, flags = 0)
            img_matches = cv2.drawMatchesKnn(frame,kp,frame,new_kp,matches,None,**draw_params)
            cv2.imshow('matches', img_matches)
            cv2.waitKey(1)'''

            src_pts = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1,1,2)
            dst_pts = np.float32([new_kp[m.trainIdx].pt for m in good]).reshape(-1,1,2)
            src_pts_hr = src_pts * siftDownsample
            dst_pts_hr = dst_pts * siftDownsample
            M, _ = cv2.findHomography(src_pts_hr, dst_pts_hr, cv2.RANSAC, maxPointsPixles)
            img3 = cv2.warpPerspective(frame_rgb_hires, np.linalg.inv(np.matmul(M,Mabs)),(firstFrame_rgb_hires.shape[1], firstFrame_rgb_hires.shape[0]))
            h, w, _ = img3.shape
            img3_cropped = img3[int(h*0.05):int(h*0.95), int(w*0.005):int(w*0.995)]
            #img3_cropped = cv2.drawKeypoints(frame, new_kp, img3_cropped, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(255,255,255))
            cv2.imwrite(path, img3_cropped)
            self.processed = self.processed + 1
            self.matched = len(good)

    def stop(self):
        self.stopped = True

def stabilize_frames(originalFolder, stabilizedFolder, captureName, videoName, segmentLengthSeconds, numberOfFrames, fps, downsample, siftDownsample, processing_threads, display=False):

    all_start_time = time.time()

    percentage = 0.00
    loaded = ''
    unloaded = '......................'

    video = originalFolder + videoName + r'.mp4'
    fvs = FileVideoStream(video).start()
    ImageProcessor = [None]*processing_threads
    for j in range(processing_threads):
        ImageProcessor[j] = ImageProcess().start()   
    frame = [None]*processing_threads
    frame_hires = [None]*processing_threads
    frame_rgb_hires = [None]*processing_threads
    h, w = [None]*processing_threads, [None]*processing_threads
    for j in range(processing_threads):
        frame[j] = fvs.read()
        h[j], w[j], _ = frame[j].shape
        frame_hires[j] = cv2.resize(frame[j],(int(w[j]/downsample),int(h[j]/downsample)))
        frame[j] = cv2.resize(frame[j],(int(w[j]/siftDownsample),int(h[j]/siftDownsample)))
        frame_rgb_hires[j] = frame_hires[j]
        frame[j] = cv2.cvtColor(frame[j], cv2.COLOR_BGR2GRAY)

    firstFrame = frame[0]
    firstFrame_rgb_hires = frame_hires[0]
   
    Mabs =  np.array([[1,0,0],[0,1,0],[0,0,1]])
    sift = cv2.xfeatures2d.SIFT_create(nfeatures = maxFeatures, nOctaveLayers = 3, contrastThreshold = contrastThresh, edgeThreshold = edgeThresh, sigma = sigmaValue)
    kp, desc = sift.detectAndCompute(firstFrame,None)


    for i in range(0, numberOfFrames+(processing_threads*2), processing_threads):

        new_frame = [None]*processing_threads
        new_frame_rgb = [None]*processing_threads
        new_frame_rgb_hires = [None]*processing_threads
        h, w = [None]*processing_threads, [None]*processing_threads
        for j in range(processing_threads):
            if i + j < numberOfFrames:
                try:
                    new_frame_rgb[j] = fvs.read()
                    h[j], w[j], _ = new_frame_rgb[j].shape
                    new_frame_rgb_hires[j] = cv2.resize(new_frame_rgb[j],(int(w[j]/downsample),int(h[j]/downsample)))
                    new_frame_rgb[j] = cv2.resize(new_frame_rgb[j],(int(w[j]/siftDownsample),int(h[j]/siftDownsample)))
                    new_frame[j] = cv2.cvtColor(new_frame_rgb[j], cv2.COLOR_BGR2GRAY)
                except:
                    break
        img_number = [None]*processing_threads
        path = [None]*processing_threads
        for j in range(processing_threads):
            
            if int((i+j)/(segmentLengthSeconds*fps)) == (i+j)/(segmentLengthSeconds*fps):
                new_kp, new_desc = sift.detectAndCompute(frame[0],None)
                FLANN_INDEX_KDTREE = 0
                index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
                search_params = dict(checks = 50)
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(desc,new_desc,k=2)
                good = []
                for m,n in matches:
                    if m.distance < nextBestMatchDistanceThresh*n.distance:
                        good.append(m)
                src_pts = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1,1,2)
                dst_pts = np.float32([new_kp[m.trainIdx].pt for m in good]).reshape(-1,1,2)
                src_pts_hr = src_pts * siftDownsample
                dst_pts_hr = dst_pts * siftDownsample
                M, _ = cv2.findHomography(src_pts_hr, dst_pts_hr, cv2.RANSAC, maxPointsPixles)
                Mabs = np.matmul(M,Mabs)
                kp, desc = sift.detectAndCompute(frame[0],None)
                      
            if i + j < numberOfFrames:
                try:
                    img_number[j] = str(i+j)
                    for k in range(5-len(img_number[j])):
                        img_number[j] = '0' + img_number[j]
                    
                    path[j] = stabilizedFolder + r'\frames\Stabilized_image_' + img_number[j] + r'.jpg'
                    ImageProcessor[j].process(firstFrame_rgb_hires, frame_rgb_hires[j], frame[j], desc, kp, Mabs, (siftDownsample/downsample), path[j])
                except:
                    break

        for j in range(processing_threads):
            if i + j < numberOfFrames:
                try:
                    frame[j] = new_frame[j]
                    frame_rgb_hires[j] = new_frame_rgb_hires[j]
                except:
                    break

        percentage = sum([ImageProcessor[x].processed for x in range(len(ImageProcessor))])/numberOfFrames*100
        cpu = psutil.cpu_percent()
        ram = dict(psutil.virtual_memory()._asdict())['percent']
        all_end_time = time.time()
        all_time = all_end_time - all_start_time
        if percentage > 100:
            percentage = 100
        loaded = '>'
        unloaded = '...................'
        for j in range(int(percentage/5)):
            unloaded = unloaded[:-1]
            loaded = '=' + loaded  
        print('\r' + 'Stabilizing Images (' + str(processing_threads) + ' threads) [' + loaded + unloaded + ']  ' + str(int(percentage*100)/100) + '%  [Features matached: ' + str(ImageProcessor[0].matched) + ' | CPU: ' + str(cpu) + '% | RAM: ' + str(ram) + '%' + ' | Total time: ' + str(round(all_time,2)) + 's]             ', end = '')
    while sum([ImageProcessor[x].Q.qsize() for x in range(len(ImageProcessor))]) > 0:
        percentage = sum([ImageProcessor[x].processed for x in range(len(ImageProcessor))])/numberOfFrames*100
        cpu = psutil.cpu_percent()
        ram = dict(psutil.virtual_memory()._asdict())['percent']
        all_end_time = time.time()
        all_time = all_end_time - all_start_time
        if percentage > 100:
            percentage = 100
        loaded = '>'
        unloaded = '...................'
        for j in range(int(percentage/5)):
            unloaded = unloaded[:-1]
            loaded = '=' + loaded  
        print('\r' + 'Stabilizing Images (' + str(processing_threads) + ' threads) [' + loaded + unloaded + ']  ' +  str(int(percentage*100)/100) + '%  [Features matached: ' + str(ImageProcessor[0].matched) + ' | CPU: ' + str(cpu) + '% | RAM: ' + str(ram) + '%' + ' | Total time: ' + str(round(all_time,2)) + 's]             ', end = '')

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
    print('\r' + 'Stabilizing Images (' + str(processing_threads) + ' threads) [====================] 100.00% ' + '. Total time: ' + str(round(all_time,2)) + 's                                                         ')




        