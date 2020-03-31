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

# Main function of the module
def track(detections_file, tracked_file, video_file, fps):

    all_start_time = time.time()

    # Initial variables
    stream = cv2.VideoCapture(video_file)
    (_, frame) = stream.read()
    _, img_w, _ = frame.shape
    vehicles_previous = []
    new_centroids = np.ones((0,0))
    tracked_vehicles = []
    tracked_vehicle_counter = 0
    detections_array = []
    temp_detections_array = []
    previous_timestamp = None
    with open(detections_file, 'r', newline='') as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            timestamp = row[0]
            if timestamp == previous_timestamp:
                temp_detections_array.append(row)
            else:
                detections_array.append(temp_detections_array)
                temp_detections_array = []
                temp_detections_array.append(row)
            previous_timestamp = timestamp
    detections_array = detections_array[2:]
    number_of_frames = len(detections_array)
    print('csv read in')

    # Loop through each seperate background segment
    for i in range(number_of_frames):

        # Initial variables for each frame 
        vehicles = []
        
        for detection in detections_array[i]:
            (_, _, x, y, w, h) = detection
            x, y, w, h = int(float(x)), int(float(y)), int(float(w)), int(float(h))
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
        legit_time = fps
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
                    if img_w*0 < vehicles[matched_centroid_index[m][1]].x < img_w*1 or i -vehicles_previous[matched_centroid_index[m][0]].startFrame < 2*fps:
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
            lost_time = fps*3
            legit_time = fps
            for v in tracked_vehicles:
                # Check that vehicle has not been seen in this frame (v.lastFrame < i) and has not been lost for too long (i-(lost_time) < v.lastFrame)
                if i-(lost_time) < v.lastFrame < i and v.visible_count > legit_time and -img_w*0.05 < v.x < img_w*1.05:                   
                    # Estimate speed in x and y
                    delta_x_avg =  int(sum([(v.trajectory[-(m+1)][0]-v.trajectory[-(m+2)][0]) for m in range(min(fps,len(v.trajectory)))]) / min(fps,len(v.trajectory)))
                    delta_y_avg =  int(sum([(v.trajectory[-(m+1)][1]-v.trajectory[-(m+2)][1]) for m in range(min(fps,len(v.trajectory)))]) / min(fps,len(v.trajectory)))
                    # Use average w and h
                    w_avg =  int(sum([v.size_history[sh_index][0] for sh_index in range(int(fps/2),len(v.size_history)-5)])/(len(v.size_history)-int(fps/2)-5))
                    h_avg =  int(sum([v.size_history[sh_index][1] for sh_index in range(int(fps/2),len(v.size_history)-5)])/(len(v.size_history)-int(fps/2)-5))
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

        # Repopulate new_centroids for this frame to included tracked, non-visible vehicles.
        new_centroids = np.ones((len(vehicles),2),dtype='float32')
        v_index = 0
        for v in vehicles:
            new_centroids[v_index,0] = int(v.x+(v.w*0.5))
            new_centroids[v_index,1] = int(v.y+(v.h*0.5))
            v_index = v_index + 1
        
        # Set vehicles_previous to current frame vehicles
        vehicles_previous = vehicles

        # Display frame
        percentage = (i/number_of_frames)*100
        loaded = '>'
        unloaded = '...................'
        for _ in range(int(percentage/5)):
            unloaded = unloaded[:-1]
            loaded = '=' + loaded  
        print('\r' + 'Tracking Detections [' + loaded + unloaded + ']  ' + str(int(percentage*100)/100) + '%            ', end = '')
    all_end_time = time.time()
    all_time = round(all_end_time - all_start_time,2)
    percentage = 1*100
    loaded = '>'
    unloaded = '...................'
    for _ in range(int(percentage/5)):
        unloaded = unloaded[:-1]
        loaded = '=' + loaded  
    print('\r' + 'Tracking Detections [' + loaded + unloaded + ']  ' + str(int(percentage*100)/100) + '% Time: ' + str(all_time) + 's            ')

    tracked_vehicles_temp = []

    for v in range(len(tracked_vehicles)):
        if tracked_vehicles[v].visible_count > fps*2:
            tracked_vehicles_temp.append(tracked_vehicles[v])

    tracked_vehicles = tracked_vehicles_temp

    w = [[(tracked_vehicles[v_index].size_history[w_index][0]) for w_index in range(len(tracked_vehicles[v_index].trajectory))] for v_index in range(len(tracked_vehicles))]
    h = [[(tracked_vehicles[v_index].size_history[h_index][1]) for h_index in range(len(tracked_vehicles[v_index].trajectory))] for v_index in range(len(tracked_vehicles))]
    x_left = [[(tracked_vehicles[v_index].startX+tracked_vehicles[v_index].trajectory[x_index][0]-int(w[v_index][x_index]/2)) for x_index in range(len(tracked_vehicles[v_index].trajectory))] for v_index in range(len(tracked_vehicles))]
    x_right = [[(tracked_vehicles[v_index].startX+tracked_vehicles[v_index].trajectory[x_index][0]+int(w[v_index][x_index]/2)) for x_index in range(len(tracked_vehicles[v_index].trajectory))] for v_index in range(len(tracked_vehicles))]
    for v_index in range(len(w)):
        w[v_index].sort()
        h[v_index].sort()
    w_avg = [int(sum([w[v_index][w_index] for w_index in range((int(len(w[v_index])/2 - 10)), (int(len(w[v_index])/2 + 10)))])/(20)) for v_index in range(len(w))]
    h_avg = [int(sum([h[v_index][h_index] for h_index in range((int(len(h[v_index])/2 - 10)), (int(len(h[v_index])/2 + 10)))])/(20)) for v_index in range(len(h))]
    x = [[(tracked_vehicles[v_index].startX+tracked_vehicles[v_index].trajectory[x_index][0]) for x_index in range(len(tracked_vehicles[v_index].trajectory))] for v_index in range(len(tracked_vehicles))]
    y = [[(tracked_vehicles[v_index].startY+tracked_vehicles[v_index].trajectory[y_index][1]) for y_index in range(len(tracked_vehicles[v_index].trajectory))] for v_index in range(len(tracked_vehicles))]
    
    for v_index in range(len(x)):
        for x_index in range(len(x[v_index])):
            if x[v_index][x_index] > img_w/2:
                x[v_index][x_index] = int(x_left[v_index][x_index]+(w_avg[v_index]/2))
            else:
                x[v_index][x_index] = int(x_right[v_index][x_index]-(w_avg[v_index]/2))

    delta_x = [[(x[v_index][x_index+4] - x[v_index][x_index]) for x_index in range(len(x[v_index])-4) ] for v_index in range(len(x))]
    delta_y = [[(y[v_index][y_index+4] - y[v_index][y_index]) for y_index in range(len(y[v_index])-4) ] for v_index in range(len(y))]
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
    
    tracked_unfiltered_file = []
    tracked_unfiltered_file.append(['timestep(s)', 'vehicle_id', 'x_centre_pos(m)', 'y_centre_pos(m)', 'length(m)', 'width(m)', 'avg_length(m)', 'avg_width(m)', 'angle(°)'])
    tracked_unfiltered_file.append([str(round(0/fps,3)), str(-1), str(1), str(1), str(1), str(1), str(1), str(1), str(1)])
    tracked_unfiltered_file.append([str(round(1/fps,3)), str(-1), str(1), str(1), str(1), str(1), str(1), str(1), str(1)])

    for i in range(number_of_frames):
        for v_index in range(len(tracked_vehicles)):
            if tracked_vehicles[v_index].startFrame < i < tracked_vehicles[v_index].lastFrame:
                f_index = i - tracked_vehicles[v_index].startFrame
                vid = v_index
                vw = round((w[v_index][f_index]),3)
                vh = round((h[v_index][f_index]),3)
                vw_avg = round((w_avg[v_index]),3)
                vh_avg = round((h_avg[v_index]),3)
                vx = round((x[v_index][f_index]),3)
                vy = round((y[v_index][f_index]),3)
                vt = round(theta[v_index][f_index],3)
                tracked_unfiltered_file.append([str(round(i/fps,3)), str(vid), str(vx), str(vy), str(vw), str(vh), str(vw_avg), str(vh_avg), str(vt)])

    tracked_unfiltered_filePath = tracked_file
    with open(tracked_unfiltered_filePath, 'w', newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerows(tracked_unfiltered_file)

track(r'D:/Connor/Autoplex/Data/Trajectories/Detections/100120-F1S1D1_DOWNSAMPLED.csv', r'D:/Connor/Autoplex/Data/Trajectories/Tracked/100120-F1S1D1_DOWNSAMPLED.csv', r'D:/Connor/Autoplex/Data/Drone_Footage/100120/100120-F1S1D1_DOWNSAMPLED.AVI', 30)
print('done')