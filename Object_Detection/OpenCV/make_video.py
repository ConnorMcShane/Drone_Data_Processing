from os import listdir
import os
from os.path import isfile, join
import cv2
import glob

def make_video(src_path, dst_video, fps, videoType = 0):
    
    percentage = 0.00
    loaded = ''
    unloaded = '......................'

    images = [img for img in os.listdir(src_path) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(src_path, images[0]))
    height, width, _ = frame.shape

    video = cv2.VideoWriter(dst_video, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(src_path, image)))
        percentage = images.index(image)/len(images)*100
        loaded = '>'
        unloaded = '...................'
        for _ in range(int(percentage/5)):
            unloaded = unloaded[:-1]
            loaded = '=' + loaded  
        print('\r' + 'Making Video ' + str(videoType) + ' [' + loaded + unloaded + ']  ' + str(int(percentage*100)/100) + '%          ', end = '')

    cv2.destroyAllWindows()
    video.release()



