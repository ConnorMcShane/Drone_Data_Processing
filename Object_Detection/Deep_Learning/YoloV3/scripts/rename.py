from os import listdir
from os import rename
from os.path import isfile, join
import cv2
mypath = r'C:\Users\cmcshan1\Documents\DroneFootage\Drone_Videos\Stabilized\images'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for filename in onlyfiles:
    newNumber = ''
    print(filename)
    newNumber = filename.split('_')[2].split('.')[0]
    if len(filename.split('_')[2].split('.')[0]) < 5:
        for i in range(5-len(filename.split('_')[2].split('.')[0])):
            newNumber = '0' + newNumber
    print(filename.replace(filename.split('_')[2].split('.')[0] ,newNumber))
    rename(join(mypath, filename), join(mypath, filename.replace(filename.split('_')[2].split('.')[0] ,newNumber))) 
