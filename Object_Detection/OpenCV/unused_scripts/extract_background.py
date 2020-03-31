import os
import sys
import cv2

def generate_background(rootFolder, videoName, method):

    stabilizedFolder = rootFolder + r'\stabilized\\' + videoName + r'\\'
    backgroundFolder = rootFolder + r'\background\\' + videoName + r'\\'
    detectionsFolder = rootFolder + r'\detections\\' + videoName + r'\\'
    originalFolder = rootFolder + r'\original\\'
    trajectoriesFolder = rootFolder +  r'\trajectories\\' + videoName + r'\\'

    folders = [rootFolder, rootFolder + r'\background\\', rootFolder + r'\stabilized\\', stabilizedFolder, stabilizedFolder + r'\videos\\', rootFolder + r'\detections\\', rootFolder +  r'\trajectories\\', backgroundFolder, detectionsFolder, originalFolder, trajectoriesFolder]
    
    video = stabilizedFolder + r'\videos\\' + videoName + r'.avi'

    for folder in folders:
        if not os.path.isdir(folder):
            os.mkdir(folder)
    
    path_relative_background = os.path.join(backgroundFolder, videoName + '_background.png')
    
    if method == 'GSOC':
        bg_subtractor = cv2.bgsegm.createBackgroundSubtractorGSOC()
    elif method == 'MOG2':
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=200, detectShadows=False)  # default history=500
    else:
        print('Undefined background subtraction method!')

    # open video
    video = cv2.VideoCapture(video)
    width = int(video.get(3))
    height = int(video.get(4))
    fps = video.get(5)
    frame_total = video.get(cv2.CAP_PROP_FRAME_COUNT)

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    frame_index = 0
    # load new frame
    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break
        frame_index = frame_index + 1

        percentage = frame_index/frame_total*100
        loaded = '>'
        unloaded = '...................'
        for j in range(int(percentage/5)):
            unloaded = unloaded[:-1]
            loaded = '=' + loaded  
        print('\r' + 'Generating background [' + loaded + unloaded + ']  ' + str(int(percentage*100)/100) + '% ', end = '')

        # cropping frame
        frame_fgmask = bg_subtractor.apply(frame)
        frame_background = bg_subtractor.getBackgroundImage()
    print('\r' + 'Generating background [====================]  ' + str(int(percentage*100)/100) + '% ')
    cv2.imwrite(path_relative_background, frame_background)

    return frame_background