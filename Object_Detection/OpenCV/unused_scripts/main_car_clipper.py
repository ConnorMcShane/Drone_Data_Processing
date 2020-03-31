import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import stabilizeFrames
import makeVideo
import perspectiveTransform
import extract_background_segmented
#import clipImages
#import trajectoryProcessing
import backgroundSubtract
import smoothTrajectories
import multiprocessing
import threading

print('Number of threads on CPU: ' + str(multiprocessing.cpu_count()) + '. Number of active threads: ' + str(threading.active_count()))

rootFolder = r'D:\Connor\Autoplex\Data\Drone\Video'
captureName = r'\100120-1403\\'
videoName = r'100120-mav1-02-1403'
segmentLengthSeconds = 20
numberOfFrames = 8700
threads = 8
fps = 29

stabilizedFolder = rootFolder + r'\Stabilized\\' + videoName + r'\\'
backgroundFolder = rootFolder + r'\Background\\' + videoName + r'\\'
detectionsFolder = rootFolder + r'\Detections\\' + videoName + r'\\'
smoothedFolder = rootFolder + r'\Smoothed\\' + videoName + r'\\'
subtractedFolder =  rootFolder + r'\Subtracted\\' + videoName + r'\\'
originalFolder = rootFolder + r'\Raw\\'
warpedFolder = rootFolder + r'\Warped\\' + videoName + r'\\'
trajectoriesFolder = rootFolder +  r'\Trajectories\\' + videoName + r'\\'

folders = [rootFolder, rootFolder + r'\Background\\', rootFolder + r'\Stabilized\\', rootFolder + r'\Subtracted\\', rootFolder + r'\Smoothed\\', smoothedFolder, subtractedFolder, subtractedFolder + r'\frames\\', subtractedFolder + r'\videos\\', smoothedFolder + r'\frames\\', smoothedFolder + r'\videos\\', stabilizedFolder, rootFolder + r'\Detections\\', rootFolder +  r'\Trajectories\\', backgroundFolder, detectionsFolder, detectionsFolder + r'\frames\\', detectionsFolder + r'\videos\\', originalFolder, trajectoriesFolder,  rootFolder + r'\Warped\\', warpedFolder, warpedFolder + r'\frames\\' , warpedFolder + r'\videos\\', stabilizedFolder + r'\frames\\' , stabilizedFolder + r'\videos\\']
videoFolders = [stabilizedFolder, warpedFolder, subtractedFolder]

for folder in folders:
    if not os.path.isdir(folder):
        os.mkdir(folder)

#stabilizeFrames.stabilize(originalFolder, stabilizedFolder, captureName, videoName, segmentLengthSeconds, numberOfFrames, fps, 1, 3, threads, display = False)
#makeVideo.makeVideo(rootFolder, videoName, fps, 0)
#perspectiveTransform.perspectiveTransform(stabilizedFolder, warpedFolder, videoName, segmentLengthSeconds, numberOfFrames, fps, 1.2, 28, -1.0e-5, threads)
#makeVideo.makeVideo(rootFolder, videoName, fps, 1)
#extract_background_segmented.generate_background(warpedFolder, backgroundFolder, videoName, segmentLengthSeconds, numberOfFrames, fps, 1)
#backgroundSubtract.subtractBackground(warpedFolder, backgroundFolder, subtractedFolder, detectionsFolder, videoName, segmentLengthSeconds, numberOfFrames, fps, 0, -0.075)
#makeVideo.makeVideo(rootFolder, videoName, fps, 2)
#makeVideo.makeVideo(rootFolder, videoName, fps, 3)
smoothTrajectories.smoothTrajectories(detectionsFolder, warpedFolder, videoName, segmentLengthSeconds, numberOfFrames, fps)