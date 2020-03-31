import time
import os

filesPath = r'D:/Connor/Autoplex/Prediction/MachineLearning/Dataset/DroneFootage/Video/Labels/backgrounds/train/'
image_files = []
label_files = []
for file in os.listdir(filesPath):
    if file.endswith(r'.jpg'):
        image_files.append(filesPath + file)
        label_files.append(filesPath + file[:-3] + 'txt')

for txt_file in label_files:
    if not os.path.exists(txt_file):
        with open(txt_file, 'w+') as f:
            f.write('')
