import time
import os
from shutil import copy

rootPath = 'D:/Connor/Autoplex/Prediction/MachineLearning/Dataset/DroneFootage/Video/Labels/'
valid_split = 0.2
testFolder = r'D:\Connor\Autoplex\Prediction\MachineLearning\Dataset\DroneFootage\Video\Labels\final_dataset\test\original'
trainFolder = r'D:\Connor\Autoplex\Prediction\MachineLearning\Dataset\DroneFootage\Video\Labels\final_dataset\train\original'

source_folders = []
for file in os.listdir(rootPath + 'sources/'):
    source_folders.append(rootPath + 'sources/' + file + '/')

for folder in source_folders:
    img_files = []
    txt_files = []
    img_file_names = []
    txt_file_names = []
    for file in os.listdir(folder):
        if file.endswith('.jpg') or file.endswith('.JPG') :
            img_files.append(os.path.join(folder, file))
            img_file_names.append(file)
        if file.endswith('.txt'):
            txt_files.append(os.path.join(folder, file))
            txt_file_names.append(file)
    if not len(img_files) == len(txt_files):
        print('fault')

    for i in range(len(txt_files)):
        if not int(i*valid_split) == i*valid_split:
            copy(txt_files[i], trainFolder)
            copy(img_files[i], trainFolder)
        if int(i*valid_split) == i*valid_split:
            copy(txt_files[i], testFolder)
            copy(img_files[i], testFolder)

print('done')
