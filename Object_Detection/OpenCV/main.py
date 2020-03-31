import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import stabilize_frames
import make_video
import perspective_transform
import extract_background_segmented
import background_subtract
import create_labels
import smooth_trajectories
import multiprocessing
import threading
from tkinter import filedialog
from tkinter import Tk
import pathlib

print('Number of threads on CPU: ' + str(multiprocessing.cpu_count()) + '. Number of active threads: ' + str(threading.active_count()))
root = Tk()
root.withdraw()
video_path = filedialog.askopenfilename(title = 'Select Video', initialdir = str(pathlib.Path(__file__).parent.absolute()))
video_name = video_path.split('/')[-1][:-4]
capture_name = '/' + video_name[:6] + '/'
original_folder = video_path[:-(len(video_name)+4)]

root_folder = filedialog.askdirectory(title = 'Select Temp Output Folde', initialdir = str(pathlib.Path(__file__).parent.absolute()))
segment_length_seconds = 10
number_of_frames = 290
threads = 20
fps = 30

stabilized_folder = root_folder + '/stabilized/' 
background_folder = root_folder + '/background/'
detections_folder = root_folder + '/detections/'
smoothed_folder = root_folder + '/smoothed/'
diff_folder = root_folder + '/difference/'
subtracted_folder =  root_folder + '/subtracted/' 
warped_folder = root_folder + '/warped/'
trajectories_folder = root_folder +  '/trajectories/'
label_folder = root_folder + '/labels/'
labels_folder = label_folder + '/labels/'
images_folder = label_folder + '/images/'
labelBBoxes_folder = label_folder + '/bounding_boxes/'

folders = [root_folder, diff_folder, diff_folder + '/frames/', diff_folder + '/videos/', smoothed_folder, subtracted_folder, subtracted_folder + '/frames/', subtracted_folder + '/videos/', smoothed_folder + '/frames/', smoothed_folder + '/videos/', stabilized_folder, background_folder, detections_folder, detections_folder + '/frames/', detections_folder + '/videos/', original_folder, trajectories_folder, warped_folder, warped_folder + '/frames/' , warped_folder + '/videos/', stabilized_folder + '/frames/' , stabilized_folder + '/videos/',  label_folder, labels_folder, images_folder, labelBBoxes_folder]
video_folders = [stabilized_folder, warped_folder, subtracted_folder]

for folder in folders:
    if not os.path.isdir(folder):
        os.mkdir(folder)

src_pt_video = stabilized_folder + video_name + '.avi'
dst_pt_video = warped_folder + video_name + '.avi'
        
stabilize_frames.stabilize_frames(original_folder, stabilized_folder, capture_name, video_name, segment_length_seconds, number_of_frames, fps, 1, 3, threads, display = False)
make_video.make_video(stabilized_folder + 'frames/', stabilized_folder + 'videos/' + video_name + '.avi', fps)
perspective_transform.perspective_transform(stabilized_folder + 'videos/' + video_name + '.avi', warped_folder + 'videos/' + video_name + '.avi', 1, number_of_frames, fps, 1.2, 28, -1.0e-5, threads)
extract_background_segmented.generate_background(warped_folder, background_folder, video_name, segment_length_seconds, number_of_frames, fps, 1, 0)
background_subtract.background_subtract(warped_folder, background_folder, subtracted_folder, diff_folder, detections_folder, video_name, segment_length_seconds, number_of_frames, fps, 0, -0.075)
create_labels.create_labels(detections_folder, warped_folder, images_folder, labels_folder, labelBBoxes_folder, number_of_frames, video_name, fps, 2, 0.2, min_size = 2500, max_size = 100000)
smooth_trajectories.smooth_trajectories(detections_folder, warped_folder, smoothed_folder, video_name, segment_length_seconds, number_of_frames, fps, createImages = True, min_size = 2500, max_size = 100000)
