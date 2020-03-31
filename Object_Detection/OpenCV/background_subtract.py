import cv2
import time
import numpy as np
import os
from threading import Thread
from queue import Queue
import csv

class FileVideoStream:
    def __init__(self, files, queueSize=32):
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
                frame = cv2.imread(self.files[self.fileNumber]) #, cv2.IMREAD_GRAYSCALE)
                self.Q.put(frame)
                self.fileNumber = self.fileNumber + 1
                if self.fileNumber >= len(self.files):
                    self.stopped = True     

    def read(self):
	    return self.Q.get()

    def stop(self):
	    self.stopped = True

# Main function of the module
def background_subtract(warpedFolder, backgroundFolder, subtractedFolder, diffFolder, detectionsFolder, videoName, segmentLengthSeconds, numberOfFrames, fps, xShadowPercentage, yShadowPercentage, min_size = 2500, max_size = 100000):
    
    all_start_time = time.time()
    dataFile = []
    untrackedUnfilteredFile = []
    untrackedUnfilteredFile.append(['timestep(s)', 'x_centre_pos', 'y_centre_pos', 'length', 'width'])
    #rawDetections = []
    pixel_to_meter = 0.024

    filesWarped = []
    for file in os.listdir(warpedFolder + r'\frames\\'):
        if file.endswith(r'.jpg'):
            filesWarped.append(os.path.join(warpedFolder + r'\frames\\', file))

    filesBackground = []
    for file in os.listdir(backgroundFolder):
        if file.endswith(r'.png'):
            filesBackground.append(os.path.join(backgroundFolder, file))

    fvs = FileVideoStream(filesWarped).start()
    img_number = 0
    loaded = ''
    unloaded = '......................'
    for j in range(int(numberOfFrames/(fps*segmentLengthSeconds))):
        background = cv2.imread(filesBackground[j], cv2.IMREAD_GRAYSCALE)
        for i in range(fps*segmentLengthSeconds):
            data = []
            img_number = img_number + 1
            frame_clr = fvs.read()
            frame = cv2.cvtColor(frame_clr, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(frame, background)
            th_delta = cv2.threshold(diff, 55, 255, cv2.THRESH_BINARY)[1]
            th_delta = cv2.GaussianBlur(th_delta,(33,33),cv2.BORDER_DEFAULT)
            th_delta = cv2.dilate(th_delta,(33,33),iterations = 3)
            th_delta = cv2.threshold(th_delta, 30, 255, cv2.THRESH_BINARY)[1]
            th_delta = cv2.GaussianBlur(th_delta,(33,33),cv2.BORDER_DEFAULT)
            th_delta = cv2.morphologyEx(th_delta, cv2.MORPH_CLOSE, (333,33))
            th_delta = cv2.threshold(th_delta, 10, 255, cv2.THRESH_BINARY)[1]

            _, contours, _ = cv2.findContours(th_delta, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            frame_contours = frame_clr
            shrinkFactor = 0.2
            cntAreas = []
            for cnt in contours:
                boundingRect = cv2.boundingRect(cnt)
                x,y,w,h = boundingRect
                vw = round((w),3)
                vh = round((h),3)
                vx = round((((x))+(vw/2)),3)
                vy = round((((y))+(vh/2)),3)
                data.append([x,y,w,h])
                untrackedUnfilteredFile.append([img_number,vx,vy,vw,vh])
                #rawDetections.append([str(round(img_number/fps,3)), round((x*pixel_to_meter,3)), round((y*pixel_to_meter,3)), round((w*pixel_to_meter,3)), round((h*pixel_to_meter,3))])
                area = w*h
                if min_size < area < max_size:
                    cntAreas.append(area)
                    shrinkFactor = 0.2

                    if xShadowPercentage > 0:
                        w = w*(1-xShadowPercentage)
                    else:
                        x = x + w*xShadowPercentage
                    
                    if yShadowPercentage > 0:
                        h = h*(1-yShadowPercentage)
                    else:
                        y = y + h*yShadowPercentage
                    
                    xs1,ys1,xs2,ys2 = int(x+(w*shrinkFactor)), int(y+(h*shrinkFactor)), int(x+(w*(1-shrinkFactor))), int(y+(h*(1-shrinkFactor)))
                    xm1,ym1,xm2,ym2 = int(x), int(y), int(x+(w)), int(y+(h))
                    xl1,yl1,xl2,yl2 = int(x-(w*shrinkFactor)), int(y-(h*shrinkFactor)), int(x+(w*(1+shrinkFactor))), int(y+(h*(1+shrinkFactor)))

                    #frame_contours = cv2.rectangle(frame_contours, (xs1,ys1), (xs2,ys2), (0,255,0), 5) # Small
                    frame_contours = cv2.rectangle(frame_contours, (xm1,ym1), (xm2,ym2), (0,255,0), 5) # Medium
                    #frame_contours = cv2.rectangle(frame_contours, (xl1,yl1), (xl2,yl2), (0,255,0), 5) # Large

            #avgArea = int(sum(cntAreas)/len(cntAreas))
            #print(avgArea)

            img_number_str = str(img_number)
            for _ in range(5-len(img_number_str)):
                img_number_str = '0' + img_number_str
            path_subtraction = subtractedFolder + r'\frames\\' + r'subtracted_image_' + img_number_str + r'.jpg'
            path_detections = detectionsFolder + r'\frames\\' + r'detections_image_' + img_number_str + r'.jpg'
            path_diff = diffFolder + r'\frames\\' + r'diff_image_' + img_number_str + r'.jpg'
            cv2.imwrite(path_subtraction, th_delta)
            cv2.imwrite(path_detections, frame_contours)
            cv2.imwrite(path_diff, diff)
            #dim = (int(frame.shape[1]/3), int(frame.shape[0]/3))
            #cv2.imshow('diff', cv2.resize(diff, dim))
            #cv2.imshow('subtraction', cv2.resize(th_delta, dim))
            #cv2.imshow('detections', cv2.resize(frame_contours, dim))
            #cv2.waitKey(1)
            dataFile.append(data)
            percentage = (img_number/numberOfFrames)*100
            loaded = '>'
            unloaded = '...................'
            for _ in range(int(percentage/5)):
                unloaded = unloaded[:-1]
                loaded = '=' + loaded  
            print('\r' + 'Subtracting Frames [' + loaded + unloaded + ']  ' + str(int(percentage*100)/100) + '%   Using Background: ' + str(j) + '       ', end = '')

    dataFile = np.asarray(dataFile)
    path_detectionsFile = detectionsFolder + videoName + r'_detections.npy'
    with open(path_detectionsFile, 'wb') as f:
        np.save(f, dataFile)

    untrackedUnfilteredFilePath = detectionsFolder + r'UntrackedUnfiltered-' + videoName  + r'.csv'
    with open(untrackedUnfilteredFilePath, 'w', newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerows(untrackedUnfilteredFile)
        
    #outputFilePath = detectionsFolder + r'raw-detections-' + videoName  + r'.csv'
    #with open(outputFilePath, 'w', newline='') as myfile:
    #    wr = csv.writer(myfile)
    #    wr.writerows(rawDetections)
    all_end_time = time.time()
    all_time = all_end_time - all_start_time
    percentage = 100
    loaded = '>'
    unloaded = '...................'
    for i in range(int(percentage/5)):
        unloaded = unloaded[:-1]
        loaded = '=' + loaded  
    print('\r' + 'Subtracting Frames [' + loaded + unloaded + ']  ' + str(int(percentage*100)/100) + '% Total time: ' + str(int(all_time)) + 's         ', end = '')