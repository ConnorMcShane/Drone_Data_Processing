import time
import os
import sys
import cv2
from threading import Thread
from queue import Queue

class FileVideoStream:
    def __init__(self, files, queueSize=8):
        self.files = files
        self.fileNumber = 0
        self.stopped = False
        self.Q = Queue(maxsize=queueSize)

    def start(self):
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
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

def generate_background(warpedFolder, backgroundFolder, videoName, segmentLengthSeconds, numberOfFrames, fps, imageStep, startBackground):
    
    all_start_time = time.time()

    files = []
    filesTemp = []
    for file in os.listdir(warpedFolder + r'\frames\\'):
        if file.endswith(r'.jpg'):
            files.append(os.path.join(warpedFolder + r'\frames\\', file))

    #fvs = FileVideoStream(files).start()
    segmentation_seconds = segmentLengthSeconds
    numberOfBackgrounds = int(numberOfFrames/(fps*segmentation_seconds))
    numberOfImages = int((fps*segmentation_seconds)/(imageStep))
    for j in range(startBackground, numberOfBackgrounds):
        startFrame = int(j*((fps*segmentation_seconds)))
        endFrame = int((j+1)*((fps*segmentation_seconds)))-1
        filesTemp = files[startFrame:endFrame:imageStep]
        img_number_str = str(j)
        for _ in range(5-len(img_number_str)):
            img_number_str = '0' + img_number_str
        path_relative_background = os.path.join(backgroundFolder, r'Background_' + img_number_str + '.png')
        bg_subtractor = cv2.bgsegm.createBackgroundSubtractorGSOC()
        #bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=numberOfImages, detectShadows=False)
        repeats = int((30/segmentLengthSeconds)*imageStep)+1
        for m in range(repeats):
            fvs = FileVideoStream(filesTemp).start()
            for i in range(len(filesTemp)):
                frame = fvs.read()
                bg_subtractor.apply(frame) 
                percentage = ((m + (i)/((fps*segmentation_seconds)))/repeats)*100
                loaded = '>'
                unloaded = '...................'
                for _ in range(int(percentage/5)):
                    unloaded = unloaded[:-1]
                    loaded = '=' + loaded  
                all_end_time = time.time()
                all_time = all_end_time - all_start_time
                print('\r' + 'Generating background [' + loaded + unloaded + '] ' + str(int(percentage*100)/100) + '% [Background number: ' + str(j) + '/' + str(numberOfBackgrounds) + ' | Between frames: ' + str(startFrame) + ' - ' + str(endFrame) + ' | Total time: ' + str(round(all_time,2)) + ']             ', end = '')
        frame_background = bg_subtractor.getBackgroundImage()
        cv2.imwrite(path_relative_background, frame_background)
    all_end_time = time.time()
    all_time = all_end_time - all_start_time
    print('All backgrounds generated. Total time: ' + str(round(all_time,2)))
