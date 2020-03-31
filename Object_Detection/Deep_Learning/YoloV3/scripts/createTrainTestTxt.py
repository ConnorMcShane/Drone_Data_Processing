import cv2
import time
import numpy as np
import os
import csv
from shutil import copyfile
import re

filesPath = r'D:\Connor\Autoplex\Prediction\MachineLearning\Dataset\DroneFootage\Video\Labels\final_dataset\test\combined'
files = []
for file in os.listdir(filesPath):
    if file.endswith(r'.jpg') or file.endswith(r'.JPG'):
        files.append('C:/Users/mlpre/Desktop/darknet-master/data/autoplex/images/' + file)

txtFile = r'C:/Users/mlpre/Desktop/darknet-master/data/autoplex/test.txt'
with open(txtFile, 'w+') as f:
    for file in files:
        f.write(file + '\n')

print('done')