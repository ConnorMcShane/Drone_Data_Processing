import cv2
import time
import numpy as np
import os
import csv
from shutil import copyfile
import re

filesPath = r'D:\Connor\Autoplex\Prediction\MachineLearning\Dataset\DroneFootage\Video\Labels\aerial-cars-dataset-master\aerial-cars-dataset-master'
files = []
for file in os.listdir(filesPath):
    if file.endswith(r'.txt'):
        files.append(os.path.join(filesPath, file))

for file in files:
    f = open(file,'r')
    text = f.read()
    text = '0' + text[1:]
    newLines = [m.start() + 1 for m in re.finditer('\n', text)]
    for i in range(len(newLines)):
        text = text[0:newLines[i]] + '0' + text[newLines[i]+1:]

    if text[len(text)-2:] == '\n0':
        text = text[0:-1]
    f.close()

    os.remove(file)
    with open(file, 'w+') as f:
        f.writelines(text)

print('done')