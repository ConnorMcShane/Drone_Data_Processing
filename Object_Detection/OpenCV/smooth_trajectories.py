import cv2
import numpy as np
from PIL import Image
import math
from scipy.spatial import distance as dist
import sys
import csv
import pickle
import os
from threading import Thread
from queue import Queue
import psutil
import threading
import time
import csv

# Define vehicle class outside of function so that it can be saved(pickled) as a file and then later loaded by another module.
class vehicle:
    def __init__(self, id, x, y, w, h, angle, visible, lastFrame, visible_count, startFrame, startX, startY, trajectory, size_history):
        self.id = id
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.angle = angle
        self.visible = visible
        self.lastFrame = lastFrame
        self.visible_count = visible_count
        self.startFrame = startFrame
        self.startX = startX
        self.startY = startY
        self.trajectory = trajectory
        self.size_history = size_history

class FileVideoStream:
    def __init__(self, files, queueSize=32):
        self.files = files
        self.fileNumber = 0
        self.stopped = False
        self.Q = Queue(maxsize=queueSize)

    def start(self):
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True
        self.t.start()
        return self
    
    def update(self):
        while True:
            if self.stopped:
                return
            if not self.Q.full():
                frame = cv2.imread(self.files[self.fileNumber])
                self.Q.put(frame)
                self.fileNumber = self.fileNumber + 1
                if self.fileNumber >= len(self.files):
                    self.stopped = True     

    def read(self):
	    return self.Q.get()

    def stop(self):
	    self.stopped = True

def rollingAverage(array, position, numberOfPoints):
    if numberOfPoints/2 > position:
        numberOfPoints = int(position*2)+1
        #print('Lower end correction. Number of points = ' + str(numberOfPoints))
    if numberOfPoints/2 > len(array)-position:
        numberOfPoints = int((len(array)-position)*2)-1
        #print('Upper end correction. Number of points = ' + str(numberOfPoints))
    rollingAverage = 0
    for i in range(numberOfPoints):
        rollingAverage = rollingAverage + array[position - int(numberOfPoints/2 - 0.5) + i]
    rollingAverage = rollingAverage/numberOfPoints
    return rollingAverage

def draw_bb(img, x, y, w, h, colour, thickness, text, font, size, text_colour):
    cv2.rectangle(img , (int(x), int(y)), (int(x+w), int(y+h)), colour, thickness)
    cv2.rectangle(img , (int(x), int(y-(24*size))), (int(x+int(len(text)*17.4*(size))), int(y)), colour, -1)
    cv2.putText(img, text, (int(x), int(y-1)), font, size, text_colour, int(thickness/2), cv2.LINE_AA)
    return img

# Main function of the module
def smooth_trajectories(detectionsFolder, warpedFolder, smoothedFolder, videoName, segmentLengthSeconds, numberOfFrames, fps, min_size = 2500, max_size = 100000, createImages = False):

    all_start_time = time.time()
    files = []
    
    for file in os.listdir(warpedFolder + r'\frames\\'):
        if file.endswith(r'.jpg'):
            files.append(os.path.join(warpedFolder + r'\frames\\', file))

    # Initial variables
    frame_rate = fps
    pixel_to_meter = 0.024
    vehicles_previous = []
    new_centroids = np.ones((0,0))
    tracked_vehicles = []
    tracked_vehicle_counter = 0

    with open(detectionsFolder + videoName + r'_detections.npy', 'rb') as f:
        detections = np.load(f, allow_pickle = True)

    # Loop through each seperate background segment
    for i in range(numberOfFrames):

        # Initial variables for each frame 
        vehicles = []

        # Load in frame and subtract the background
        new_frame = cv2.imread(files[0])
        
        for detection in detections[i]:
            (x, y, w, h) = detection
            # Set object tracking boundry to only include the road and ignore the very edge of the image
            if new_frame.shape[0]*0.01 < y < new_frame.shape[0]*0.99 and new_frame.shape[1]*0.01 < x and x + w < new_frame.shape[1]*0.99:
                if  min_size < (w*h) < max_size:
                    # Add to current frame vahicle array
                    vehicles.append(vehicle((len(vehicles)+tracked_vehicle_counter+2), x, y, w, h, 0, True, i, 1, i, x, y, [], []))

        # CENTROID TRACKING
        # Set previous frame centroids to 'centroids' variable
        centroids = new_centroids
        # Make blank np array to hold this frames vehicle center poinrs
        new_centroids = np.ones((len(vehicles),2),dtype='float32')
        # Loop through each vehicle in current frame and add centroid x and y points to new_centroid array
        v_index = 0
        for v in vehicles:
            new_centroids[v_index,0] = int(v.x+(v.w*0.5))
            new_centroids[v_index,1] = int(v.y+(v.h*0.5))
            v_index = v_index + 1
        
        # Clear matched centroid arrays
        matched_centroid_ids = []
        matched_centroid_index = []
        legit_time = frame_rate*2
        if centroids.size != 0 and new_centroids.size != 0:
            # Calculate distance between all centroids and add to D array
            D = dist.cdist(centroids, new_centroids)
            # Loop through each row of D array and find minimum distance
            for d_row in range(D.shape[0]):
                # Filter out min distances over threshold value
                maxDistance =  50
                if np.amin(D[d_row]) < maxDistance: # and vehicles_previous[d_row].w*vehicles_previous[d_row].h*0.8 < (vehicles[np.where(D[d_row] == np.amin(D[d_row]))[0][0]].w)*(vehicles[np.where(D[d_row] == np.amin(D[d_row]))[0][0]].h) < vehicles_previous[d_row].w*vehicles_previous[d_row].h*1.2:
                    src_vid = vehicles_previous[d_row].id
                    dst_vid = vehicles[np.where(D[d_row] == np.amin(D[d_row]))[0][0]].id
                    src_vindex = d_row
                    dst_vindex = np.where(D[d_row] == np.amin(D[d_row]))[0][0]
                    #if vehicles[dst_vindex].w*0.8 < vehicles_previous[src_vindex].w < vehicles[dst_vindex].w*1.2:
                    if dst_vid in [matched_centroid_ids[m][1] for m in range(len(matched_centroid_ids))]:
                        previousScrRow = matched_centroid_index[[matched_centroid_ids[m][1] for m in range(len(matched_centroid_ids))].index(dst_vid)][0]
                        if np.amin(D[d_row]) < np.amin(D[previousScrRow]) or (vehicles_previous[d_row].visible_count > vehicles_previous[previousScrRow].visible_count and vehicles_previous[previousScrRow].visible_count < legit_time and np.amin(D[d_row])< maxDistance):
                            matched_centroid_index.remove(matched_centroid_index[[matched_centroid_ids[m][1] for m in range(len(matched_centroid_ids))].index(dst_vid)])
                            matched_centroid_ids.remove(matched_centroid_ids[[matched_centroid_ids[m][1] for m in range(len(matched_centroid_ids))].index(dst_vid)])
                            matched_centroid_ids.append([src_vid,dst_vid])
                            matched_centroid_index.append([src_vindex,dst_vindex])
                    else:
                        matched_centroid_ids.append([src_vid, dst_vid])
                        matched_centroid_index.append([src_vindex, dst_vindex])
                

            # Set new vehicle counter to 0 (vehicles that have not been seen in previous frame)
            new_vehicle_counter = 0

            # Loop through each matched pair of centroids
            for m in range(len(matched_centroid_ids)):

                # Check if the previous vehicle Id is greater than the list of tracked vehicles
                if int(vehicles_previous[matched_centroid_index[m][0]].id) > tracked_vehicle_counter:        
                    # Assign new id
                    vehicles[matched_centroid_index[m][1]].id = tracked_vehicle_counter + new_vehicle_counter + 1
                    # Begin trajectory logging
                    vehicles[matched_centroid_index[m][1]].trajectory.append([0, 0])
                    # Add current bounding box size to size history array
                    vehicles[matched_centroid_index[m][1]].size_history.append([vehicles[matched_centroid_index[m][1]].w, vehicles[matched_centroid_index[m][1]].h])
                    # Add vehicle to list of tracked vehicles
                    tracked_vehicles.append(vehicles[matched_centroid_index[m][1]])
                    # Increment new vehicle counter
                    new_vehicle_counter = new_vehicle_counter + 1

                # else = matches with a previous vehicle id that is already part of the tracked vehicle array
                else:
                    # Check is vehicle has been visible in previous frame or if it has been refound in this frame
                    if vehicles_previous[matched_centroid_index[m][0]].visible == False:
                        # Vehicle found. Re-assining total visible count to found vehicle
                        vehicles[matched_centroid_index[m][1]].visible_count = vehicles_previous[matched_centroid_index[m][0]].visible_count + 1
                    else:
                        # Vehicle was not lost. Increment total visible count by 1
                        vehicles[matched_centroid_index[m][1]].visible_count = vehicles_previous[matched_centroid_index[m][0]].visible_count + 1

                    vehicles[matched_centroid_index[m][1]].id = vehicles_previous[matched_centroid_index[m][0]].id
                    vehicles[matched_centroid_index[m][1]].startFrame = vehicles_previous[matched_centroid_index[m][0]].startFrame
                    vehicles[matched_centroid_index[m][1]].startX = vehicles_previous[matched_centroid_index[m][0]].startX
                    vehicles[matched_centroid_index[m][1]].startY = vehicles_previous[matched_centroid_index[m][0]].startY
                    vehicles[matched_centroid_index[m][1]].trajectory = vehicles_previous[matched_centroid_index[m][0]].trajectory
                    vehicles[matched_centroid_index[m][1]].size_history = vehicles_previous[matched_centroid_index[m][0]].size_history

                    # Update trajectory and size history with new values
                    delta_x = vehicles[matched_centroid_index[m][1]].x - vehicles_previous[matched_centroid_index[m][0]].startX
                    delta_y = vehicles[matched_centroid_index[m][1]].y - vehicles_previous[matched_centroid_index[m][0]].startY
                    vehicles[matched_centroid_index[m][1]].trajectory.append([delta_x, delta_y])
                    vehicles[matched_centroid_index[m][1]].size_history.append([vehicles[matched_centroid_index[m][1]].w, vehicles[matched_centroid_index[m][1]].h])
                    
                    # Set last frame and visibility
                    vehicles[matched_centroid_index[m][1]].visible = True
                    vehicles[matched_centroid_index[m][1]].lastFrame = i

                    # Set tracked vehichle to new updated current frame vehicle
                    tracked_vehicles[int(vehicles_previous[matched_centroid_index[m][0]].id)-1] = vehicles[matched_centroid_index[m][1]] 

            # Update tracked vehicle counter with all new vehicles added to array in current frame
            tracked_vehicle_counter = tracked_vehicle_counter + new_vehicle_counter

            # Deal with of unmatched vehicles
            lost_time = frame_rate*3
            legit_time = frame_rate*1
            for v in tracked_vehicles:
                # Check that vehicle has not been seen in this frame (v.lastFrame < i) and has not been lost for too long (i-(lost_time) < v.lastFrame)
                if i-(lost_time) < v.lastFrame < i and v.visible_count > legit_time and new_frame.shape[1]*0.05 < v.x < new_frame.shape[1]*0.95:                   
                    # Estimate speed in x and y
                    delta_x_avg =  int(sum([(v.trajectory[-(m+1)][0]-v.trajectory[-(m+2)][0]) for m in range(min(fps,len(v.trajectory)))]) / min(fps,len(v.trajectory)))
                    delta_y_avg =  int(sum([(v.trajectory[-(m+1)][1]-v.trajectory[-(m+2)][1]) for m in range(min(fps,len(v.trajectory)))]) / min(fps,len(v.trajectory)))
                    # Use average w and h
                    w_avg =  int(sum([v.size_history[sh_index][0] for sh_index in range(int(frame_rate/2),len(v.size_history)-5)])/(len(v.size_history)-int(frame_rate/2)-5))
                    h_avg =  int(sum([v.size_history[sh_index][1] for sh_index in range(int(frame_rate/2),len(v.size_history)-5)])/(len(v.size_history)-int(frame_rate/2)-5))
                    # Update x and y position
                    v.x = v.x + delta_x_avg + int((v.w - w_avg)/2)
                    v.y = v.y + delta_y_avg + int((v.h - h_avg)/2)
                    v.w =  w_avg
                    v.h =  h_avg

                    # Calculate absolute x and y changes from x and y start
                    delta_x = v.x - v.startX
                    delta_y = v.y - v.startY
                    # Append predictions to trajectory and size history
                    v.trajectory.append([delta_x, delta_y])
                    v.size_history.append([v.w, v.h])

                    # Update visibility
                    v.visible = False

                    # Add to list of vehicles in scene
                    vehicles.append(v)

        # Repopulate new_centroids for this frame.... not sure why this is nessasary or why I added it here but it is. Will try and remember and update note.
        new_centroids = np.ones((len(vehicles),2),dtype='float32')
        v_index = 0
        for v in vehicles:
            new_centroids[v_index,0] = int(v.x+(v.w*0.5))
            new_centroids[v_index,1] = int(v.y+(v.h*0.5))
            v_index = v_index + 1
        
        # Set vehicles_previous to current frame vehicles
        vehicles_previous = vehicles

        # Display frame
        percentage = (i/numberOfFrames)*100
        loaded = '>'
        unloaded = '...................'
        for _ in range(int(percentage/5)):
            unloaded = unloaded[:-1]
            loaded = '=' + loaded  
        print('\r' + 'Tracking Detections [' + loaded + unloaded + ']  ' + str(int(percentage*100)/100) + '%            ', end = '')
        show = False 
        if show == True:
            for v in vehicles:
                if v.visible == True:
                    new_frame = draw_bb(new_frame, v.x, v.y, v.w, v.h, (150, 255, 200), 5, 'Car:' + str(v.id), 0, 1, (0, 0, 0))
                else:
                    new_frame = draw_bb(new_frame, v.x, v.y, v.w, v.h, (255, 255, 200), 5, 'Car:' + str(v.id), 0, 1, (0, 0, 0))
            dim = (int(new_frame.shape[1]/4), int(new_frame.shape[0]/4))
            cv2.imshow('tracked', cv2.resize(new_frame, dim))
            cv2.waitKey(1)
    all_end_time = time.time()
    all_time = round(all_end_time - all_start_time,2)
    percentage = 1*100
    loaded = '>'
    unloaded = '...................'
    for _ in range(int(percentage/5)):
        unloaded = unloaded[:-1]
        loaded = '=' + loaded  
    print('\r' + 'Tracking Detections [' + loaded + unloaded + ']  ' + str(int(percentage*100)/100) + '% Time: ' + str(all_time) + 's            ')

    #Smooth vehicle trajectories
    all_start_time = time.time()
    rollingAverageSampleSize = 5

    tracked_vehicles_temp = []

    for v in range(len(tracked_vehicles)):
        if tracked_vehicles[v].visible_count > fps*2:
            tracked_vehicles_temp.append(tracked_vehicles[v])

    tracked_vehicles = tracked_vehicles_temp

    x = [[(tracked_vehicles[v_index].startX+tracked_vehicles[v_index].trajectory[x_index][0]) for x_index in range(len(tracked_vehicles[v_index].trajectory))] for v_index in range(len(tracked_vehicles))]
    y = [[(tracked_vehicles[v_index].startY+tracked_vehicles[v_index].trajectory[y_index][1]) for y_index in range(len(tracked_vehicles[v_index].trajectory))] for v_index in range(len(tracked_vehicles))]
    w = [[(tracked_vehicles[v_index].size_history[w_index][0]) for w_index in range(len(tracked_vehicles[v_index].trajectory))] for v_index in range(len(tracked_vehicles))]
    h = [[(tracked_vehicles[v_index].size_history[h_index][1]) for h_index in range(len(tracked_vehicles[v_index].trajectory))] for v_index in range(len(tracked_vehicles))]
    for v_index in range(len(w)):
        w[v_index].sort()
        h[v_index].sort()
    w_avg = [int(sum([w[v_index][w_index] for w_index in range((int(len(w[v_index])/2 - 10)), (int(len(w[v_index])/2 + 10)))])/(20)) for v_index in range(len(w))]
    h_avg = [int(sum([h[v_index][h_index] for h_index in range((int(len(h[v_index])/2 - 10)), (int(len(h[v_index])/2 + 10)))])/(20)) for v_index in range(len(h))]

    trackedUnfilteredFile = []
    trackedUnfilteredFile.append(['timestep(s)', 'vehicle_id', 'x_centre_pos(m)', 'y_centre_pos(m)', 'length(m)', 'width(m)'])

    for i in range(numberOfFrames):
        for v_index in range(len(tracked_vehicles)):
            if tracked_vehicles[v_index].startFrame < i < tracked_vehicles[v_index].lastFrame:
                f_index = i - tracked_vehicles[v_index].startFrame
                vid = v_index #tracked_vehicles[v_index].id
                vw = round((w_avg[v_index])*pixel_to_meter,3)
                vh = round((h_avg[v_index])*pixel_to_meter,3)
                vx = round((((x[v_index][f_index])*pixel_to_meter)+(vw/2)),3)
                vy = round((((y[v_index][f_index])*pixel_to_meter)+(vh/2)),3)
                trackedUnfilteredFile.append([str(round(i/fps,3)), str(vid), str(vx), str(vy), str(vw), str(vh)])

    trackedUnfilteredFilePath = smoothedFolder + r'TrackedUnfiltered-' + videoName  + r'.csv'
    with open(trackedUnfilteredFilePath, 'w', newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerows(trackedUnfilteredFile)
    
    rollingAverageSampleSize = 15

    x_ravg = [[int(rollingAverage(x[v_index], x_index, rollingAverageSampleSize)) for x_index in range(len(x[v_index]))] for v_index in range(len(x))]
    y_ravg = [[int(rollingAverage(y[v_index], y_index, rollingAverageSampleSize)) for y_index in range(len(y[v_index]))] for v_index in range(len(y))]

    delta_x = [[(x_ravg[v_index][x_index+4] - x_ravg[v_index][x_index]) for x_index in range(len(x_ravg[v_index])-4) ] for v_index in range(len(x_ravg))]
    delta_y = [[(y_ravg[v_index][y_index+4] - y_ravg[v_index][y_index]) for y_index in range(len(y_ravg[v_index])-4) ] for v_index in range(len(y_ravg))]
    for v_index in range(len(delta_x)):
        delta_x[v_index].insert(0, delta_x[v_index][0]) 
        delta_x[v_index].insert(0, delta_x[v_index][0]) 
        delta_y[v_index].insert(0, delta_y[v_index][0]) 
        delta_y[v_index].insert(0, delta_y[v_index][0]) 
        delta_x[v_index].append(delta_x[v_index][-1])
        delta_x[v_index].append(delta_x[v_index][-1])
        delta_y[v_index].append(delta_y[v_index][-1])
        delta_y[v_index].append(delta_y[v_index][-1])
    
    theta = []
    for v_index in range(len(delta_x)):
        theta.append([])
        for t_index in range(len(delta_x[v_index])):
            if delta_x[v_index][t_index] >= 0 and delta_y[v_index][t_index] >= 0:
                if (delta_x[v_index][t_index]) == 0:
                    theta[v_index].append(0)
                else:
                    theta[v_index].append(90-math.degrees(math.atan((delta_y[v_index][t_index])/(delta_x[v_index][t_index]))))
            elif delta_x[v_index][t_index] >= 0 and delta_y[v_index][t_index] < 0:
                if (delta_x[v_index][t_index]) == 0:
                    theta[v_index].append(180)
                else:
                    theta[v_index].append(90+math.degrees(math.atan((-delta_y[v_index][t_index])/(delta_x[v_index][t_index]))))
            elif delta_x[v_index][t_index] < 0 and delta_y[v_index][t_index] < 0:
                theta[v_index].append(270-math.degrees(math.atan((-delta_y[v_index][t_index])/(-delta_x[v_index][t_index]))))
            elif delta_x[v_index][t_index] < 0 and delta_y[v_index][t_index] >= 0:
                theta[v_index].append(270+math.degrees(math.atan((delta_y[v_index][t_index])/(-delta_x[v_index][t_index]))))

    outputFile = []
    outputFile.append(['timestep(s)', 'vehicle_id', 'x_centre_pos(m)', 'y_centre_pos(m)', 'length(m)', 'width(m)', 'theta(deg)'])
    if createImages == True:
        fvs2 = FileVideoStream(files).start()
    for i in range(numberOfFrames):
        if createImages == True:
            new_frame = fvs2.read()
        for v_index in range(len(tracked_vehicles)):
            if tracked_vehicles[v_index].startFrame < i < tracked_vehicles[v_index].lastFrame:
                f_index = i - tracked_vehicles[v_index].startFrame
                vid = v_index #tracked_vehicles[v_index].id
                if createImages == True:
                    new_frame = draw_bb(new_frame, (x_ravg[v_index][f_index]), (y_ravg[v_index][f_index]), w_avg[v_index], h_avg[v_index], (255, 255, 200), 5, 'Car:' + str(vid), 0, 1, (0, 0, 0))
                vw = round((w_avg[v_index])*pixel_to_meter,3)
                vh = round((h_avg[v_index])*pixel_to_meter,3)
                vx = round((((x_ravg[v_index][f_index])*pixel_to_meter)+(vw/2)),3)
                vy = round((((y_ravg[v_index][f_index])*pixel_to_meter)+(vh/2)),3)
                vtheta = round(theta[v_index][f_index],2)
                outputFile.append([str(round(i/fps,3)), str(vid), str(vx), str(vy), str(vw), str(vh), str(vtheta)])
        percentage = (i/numberOfFrames)*100
        loaded = '>'
        unloaded = '...................'
        for _ in range(int(percentage/5)):
            unloaded = unloaded[:-1]
            loaded = '=' + loaded  
        print('\r' + 'Smoothing Detections [' + loaded + unloaded + ']  ' + str(int(percentage*100)/100) + '%             ', end = '')
        show = False
        if createImages == True:
            if show == True:
                dim = (int(new_frame.shape[1]/4), int(new_frame.shape[0]/4))
                cv2.imshow('smoothed', cv2.resize(new_frame, dim))
                cv2.waitKey(1)
            img_number_str = str(i)
            for _ in range(5-len(img_number_str)):
                img_number_str = '0' + img_number_str
            path_smoothed = smoothedFolder + r'\frames\\' + r'Smoothed_image_' + img_number_str + r'.jpg'
            cv2.imwrite(path_smoothed, new_frame)
  
    outputFilePath = smoothedFolder + r'Detections-' + videoName  + r'.csv'
    with open(outputFilePath, 'w', newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerows(outputFile)

    all_end_time = time.time()
    all_time = round(all_end_time - all_start_time, 2)
    percentage = 1*100
    loaded = '>'
    unloaded = '...................'
    for _ in range(int(percentage/5)):
        unloaded = unloaded[:-1]
        loaded = '=' + loaded  
    print('\r' + 'Smoothing Detections [' + loaded + unloaded + ']  ' + str(int(percentage*100)/100) + '% Time: ' + str(all_time) + 's            ')

