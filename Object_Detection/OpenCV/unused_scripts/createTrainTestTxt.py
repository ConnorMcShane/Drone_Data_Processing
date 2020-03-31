import cv2
import time
import numpy as np
import os
import csv
from shutil import copyfile
import re

filesPath = r'C:\Users\mlpre\Downloads\yolov3-master\yolov3-master\data\autoplex\images'
files = []
for file in os.listdir(filesPath):
    if file.endswith(r'.jpg'):
        files.append('C:/Users/mlpre/Downloads/yolov3-master/yolov3-master/data/autoplex/images/' + file)

for file in files:
    txtFile = file[:-3] + 'txt'
    if not os.path.exists(txtFile):
        with open(txtFile, 'w+') as f:
            f.write('')

#txtFile = r'C:\Users\mlpre\darknet-master\data\autoplex\test.txt'
#with open(txtFile, 'w+') as f:
#    for file in files:
#        f.write(file + '\n')

print('done')