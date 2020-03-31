import cv2
import time
import numpy as np
import os
import csv
from shutil import copyfile
import re

filesPath = r'D:/Connor/Autoplex/Prediction/MachineLearning/Dataset/DroneFootage/Video/Labels/sources/aerial-cars-dataset-master/'
files = []
for file in os.listdir(filesPath):
    if file.endswith(r'.png'):
        files.append(os.path.join(filesPath, file))

for file in files:
    
    cv2.imwrite(file[0:-3] + 'jpg', cv2.imread(file))

    os.remove(file)

print('done')