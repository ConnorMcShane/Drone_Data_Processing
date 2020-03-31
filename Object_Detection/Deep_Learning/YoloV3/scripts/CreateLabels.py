import cv2
import time
import numpy as np
import os
import csv
from shutil import copyfile

def CreateLabels(detectionsFolder, warpedFolder, imagesFolder, labelsFolder, bboxFolder, numberOfFrames, videoName, fps, segmentLengthSeconds, val_split, min_size = 2500, max_size = 100000):

    with open(detectionsFolder + videoName + r'_detections.npy', 'rb') as f:
            detections = np.load(f, allow_pickle = True)

    images = []
    for file in os.listdir(warpedFolder + r'\frames\\'):
        if file.endswith(r'.jpg'):
            images.append(os.path.join(warpedFolder + r'\frames\\', file))
    bboxes = []
    for file in os.listdir(detectionsFolder + r'\frames\\'):
        if file.endswith(r'.jpg'):
            bboxes.append(os.path.join(detectionsFolder + r'\frames\\', file))
    image = cv2.imread(images[0])
    imgH = image.shape[0]
    imgW = image.shape[1]

    #val_numb = int(numberOfFrames*val_split)
    #train_numb = numberOfFrames - val_numb

    #folders = [imagesFolder + r'val\\', imagesFolder + r'train\\', labelsFolder + r'val\\', labelsFolder + r'train\\']
    #for folder in folders:
    #if not os.path.isdir(folder):
    #    os.mkdir(folder)
    
    for i in range(0, numberOfFrames, fps*segmentLengthSeconds):
        imgNumb = str(i)
        image = cv2.imread(images[i])
        for _ in range(5-len(imgNumb)):
            imgNumb = '0' + imgNumb
        labels = []
        labelsPath = labelsFolder + videoName + r'_image_' + imgNumb + r'.txt'
        imagePath = imagesFolder + videoName + r'_image_' + imgNumb + r'.jpg'
        bboxPath = bboxFolder + r'BBox_image_' + imgNumb + r'.jpg'
        #labelsPath = labelsFolder + r'train\\' + videoName + r'_image_' + imgNumb + r'.txt'
        #imagePath = imagesFolder + r'train\\' + videoName + r'_image_' + imgNumb + r'.jpg'
        #if i > train_numb:
        #    labelsPath = labelsFolder + r'val\\' + r'Warped_image_' + imgNumb + r'.txt'
        #    imagePath = imagesFolder + r'val\\' + r'Warped_image_' + imgNumb + r'.jpg'
        for detection in detections[i]:
            (x, y, w, h) = detection
            if  min_size < (w*h) < max_size and w < h*3 and h < w:
                #if w < 250:
                image = cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 3)
                (x, y, w, h) = (round(((x+(w/2)))/imgW, 6), round((y+(h/2))/imgH, 6), round(w/imgW, 6), round(h/imgH, 6))
                labels.append('0 ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + '\n')
                #else:
                #    image = cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,255), 3)
                #    (x, y, w, h) = (round(((x+(w/2)))/imgW, 6), round((y+(h/2))/imgH, 6), round(w/imgW, 6), round(h/imgH, 6))
                #    labels.append('1 ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + '\n')                    
        finish = False
        while not finish:
            
            cv2.imshow('bboxes', cv2.resize(image, (int(imgW/3), int(imgH/3))))
            key = cv2.waitKey(0)
            if key == ord('y'):

                cv2.imwrite(bboxPath, image)
                copyfile(images[i], imagePath)
                
                with open(labelsPath,"w") as labelsFile:
                    labelsFile.writelines(labels) # for L = labels
                labelsFile.close()
                finish = True

            if key == ord('n'):
                finish = True
        

        print('\r' + 'Creating labels ' + str(round((i+1)/numberOfFrames*100,2)) + '%', end = '')
    print('\r' + 'Creating labels ' + str(100.00) + '%')