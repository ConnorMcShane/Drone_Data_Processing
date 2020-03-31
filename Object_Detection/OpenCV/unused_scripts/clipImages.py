import cv2
import numpy as np
from PIL import Image
import math
from scipy.spatial import distance as dist
import sys
import csv
import pickle
import os

# Define vehicle class outside of function so that it can be saved(pickled) as a file and then later loaded by another module.
class vehicle:
    def __init__(self, id, x, y, w, h, angle, speedx, speedy, speed, visible, lastFrame, visible_count, startFrame, startX, startY, trajectory, size_history):
        self.id = id
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.angle = angle
        self.speedx = speedx
        self.speedy = speedy
        self.speed = speed
        self.visible = visible
        self.lastFrame = lastFrame
        self.visible_count =visible_count
        self.startFrame = startFrame
        self.startX = startX
        self.startY = startY
        self.trajectory = trajectory
        self.size_history = size_history

# Main function of the module
def clipImages(rootFolder, videoName, segmentLengthSeconds, numberOfFrames, fps, trySplit, min_contour, max_contour, threshold_value, show, src=0):
    
    stabilizedFolder = rootFolder + r'\stabilized\\' + videoName + r'\\'
    backgroundFolder = rootFolder + r'\background\\' + videoName + r'\\'
    detectionsFolder = rootFolder + r'\detections\\' + videoName + r'\\'
    originalFolder = rootFolder + r'\original\\'
    warpedFolder = rootFolder + r'\warped\\' + videoName + r'\\'
    trajectoriesFolder = rootFolder +  r'\trajectories\\' + videoName + r'\\'

    folders = [rootFolder, rootFolder + r'\background\\', rootFolder + r'\stabilized\\', stabilizedFolder, rootFolder + r'\detections\\', rootFolder +  r'\trajectories\\', backgroundFolder, detectionsFolder, detectionsFolder + r'\frames\\', detectionsFolder + r'\videos\\', originalFolder, trajectoriesFolder,  rootFolder + r'\warped\\', warpedFolder, warpedFolder + r'\frames\\' , warpedFolder + r'\videos\\']

    video = warpedFolder + r'\videos\\' + videoName + r'.avi'
    if src == 1:
        video = originalFolder + videoName + r'.avi'

    for folder in folders:
        if not os.path.isdir(folder):
            os.mkdir(folder)
    
    cap = cv2.VideoCapture(video)
    kernel = np.ones((3,3),np.uint8)
    kernelErode = np.ones((2,2),np.uint8)
    kernelDilate = np.ones((9,9),np.uint8)
    kernelClose = np.ones((5,5),np.uint8)
    kernelOpen = np.ones((5,5),np.uint8)
    kernelCloseFinal = np.ones((3,3),np.uint8)
    
    # Function to draw transparent bounding boxes
    def draw_bb_transparent(img, x, y, w, h, colour, thickness, text, font, size, text_colour):
        overlay = img.copy()
        alpha = 0.7
        cv2.rectangle(overlay , (x, y), (x+w, y+h), colour, thickness)
        cv2.rectangle(overlay , (x, y-8), (x+int(len(text)*5.8), y), colour, -1)
        cv2.putText(overlay, text, (x, y-1), font, size, text_colour, 1, cv2.LINE_AA)
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        return img

    # Function to draw bounding boxes
    def draw_bb(img, x, y, w, h, colour, thickness, text, font, size, text_colour):
        cv2.rectangle(img , (x, y), (x+w, y+h), colour, thickness)
        cv2.rectangle(img , (x, y-8), (x+int(len(text)*5.8), y), colour, -1)
        cv2.putText(img, text, (x, y-1), font, size, text_colour, 1, cv2.LINE_AA)
        return img

    # Initial variables
    segmentation_seconds = segmentLengthSeconds
    frame_rate = fps
    frameStep = int(segmentation_seconds/4)
    frame_index = 0
    vehicles_previous = []
    new_centroids = np.ones((0,0))
    tracked_vehicles = []
    tracked_vehicle_counter = 0
    number_of_speeds = int(frame_rate*1)
    pixel_to_meter = 50/7.5
    mps_mph = 2.237
    laneWidth = 53
    frame_counter = 0 
    superposeQty = 7

    # Loop through each seperate background segment
    for j in range(int(numberOfFrames/(frame_rate*segmentation_seconds))):
        # Load in background for segment
        background = cv2.imread(backgroundFolder + r'\\' + videoName + r'_background_' + str(j) + r'.png', cv2.IMREAD_GRAYSCALE)
        print('Using background: ' + str(j))
        # Loop through each frame in background segment
        for i in range((j*frame_rate*segmentation_seconds),((j+1)*(frame_rate*segmentation_seconds)),frameStep):
            if i + superposeQty < numberOfFrames:
                # Initial variables for each frame 
                vehicles = []
                car_found = False
                frame_counter = frame_counter + 1
                font = 6

                # Load in frame and subtract the background
                cap.set(1, i)
                ret, new_frame = cap.read()
                new_frame_clr = new_frame
                new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(new_frame, background)

                # Morphology operations
                th_delta = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)[1]
                thresh = th_delta
                th_delta = cv2.GaussianBlur(th_delta,(33,33),cv2.BORDER_DEFAULT)
                th_delta = cv2.dilate(th_delta,kernel,iterations = 3)          
                th_delta = cv2.threshold(th_delta, 125, 255, cv2.THRESH_BINARY)[1]

                # Find contours and filter on minimum size to pick out cars
                a, contours, hierarchy = cv2.findContours(th_delta, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                for contour in contours:
                    (x, y, w, h) = cv2.boundingRect(contour)
                    # Set object tracking boundry to only include the road and ignore the very edge of the image
                    if new_frame.shape[0]*0.01 < y < new_frame.shape[0]*0.99 and new_frame.shape[1]*0.01 < x and x + w < new_frame.shape[1]*0.99:
                        if  min_contour < cv2.contourArea(contour) < max_contour:
                            # Get angle
                            (x,y),(width,height),theta = cv2.minAreaRect(contour)
                            rotatedRect = cv2.minAreaRect(contour)
                            # Define contour bounding rectangle
                            (x, y, w, h) = cv2.boundingRect(contour)
                            # Shadow Correction
                            (xsc, ysc, wsc, hsc) = (int(x+(0.5*h)), y, int(w-(0.7*h)), h) 
                            (xsc, ysc, wsc, hsc) = (x, y, w, h)
                            # Filter our suspicously tall vehicles
                            #if ((h < 50 or w < 50) or cv2.contourArea(contour) > h*w*0.75) and cv2.contourArea(contour) > h*w*0.1:
                            if (cv2.contourArea(contour) > h*w*0.5 and (h < laneWidth or w < laneWidth)) or cv2.contourArea(contour) > h*w*0.8:
                                # Add to current frame vahicle array
                                vehicles.append(vehicle(len(vehicles)+tracked_vehicle_counter+2, xsc, ysc, wsc, hsc, 0, [None]*number_of_speeds, [None]*number_of_speeds, 0, True, i, 1, i,xsc,ysc,[],[]))
                                box = cv2.boxPoints(rotatedRect) 
                                box = np.int0(box)
                                #cv2.drawContours(new_frame_clr,[box],0,(0,0,255),2)
                                #cv2.rectangle(new_frame_clr , (x, y), (x+w, y+h), colour, thickness)
                            else:
                                if trySplit == True:
                                    # Split up into verticle segments to check if contour is made up of 2 cars
                                    
                                    segment_number = int(w/10)
                                    
                                    middle_sum = 0
                                    counter = 0
                                    lower_x_start = 0
                                    lower_x_end = 0
                                    upper_x_start = 0
                                    upper_x_end = 0
                                    sw = w/segment_number
                                    for k in range(segment_number):
                                        # Extract individual verticle segments and dilate + close them to increase white space. Then draw bounding boxes of contours within segments
                                        segment = th_delta[y:y+h,int(x+((sw)*(k))):int(x+((sw)*(k+1)))]
                                        segment = cv2.dilate(segment,kernel,iterations = 3)
                                        segment = cv2.morphologyEx(segment, cv2.MORPH_CLOSE, kernel, iterations = 3)
                                        sa, segment_contours, segment_hierarchy = cv2.findContours(segment, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                        c = max(segment_contours, key = cv2.contourArea)
                                        (sx, sy, sw, sh) = cv2.boundingRect(segment_contours[0])
                                        # Draw each segment
                                        cv2.rectangle(new_frame_clr , (int(x+((sw)*(k))), y+sy), (int(x+((sw)*(k+1))), y+sy+sh), (150, 150, 255), 2)
                                        
                                        # Sum all middle boundries of contour bounding boxes for segments
                                        if h*0.25 < sy < h*0.75:
                                            middle_sum = middle_sum +sy
                                            counter = counter + 1
                                        if h*0.25 < sy + sh < h*0.75:
                                            middle_sum = middle_sum +sy +sh
                                            counter = counter + 1
                                        # Find the starting and ending(x axis) segments for upper and lower car
                                        if sy < h*0.10:
                                            upper_x_end = k
                                            if upper_x_start == 0:
                                                upper_x_start = k
                                        if sy + sh > h*0.9:
                                            lower_x_end = k
                                            if lower_x_start == 0:
                                                lower_x_start = k
                                    if counter > 0:
                                        middle_average = middle_sum/counter
                                        xu1 = int(x + sw*upper_x_start)
                                        xu2 = int(x + sw*upper_x_end)
                                        yu1 = y
                                        yu2 = int(y + middle_average)
                                        xl1 = int(x + sw*lower_x_start)
                                        xl2 = int(x + sw*lower_x_end)
                                        yl1 = int(y + middle_average)
                                        yl2 = int(y + h)
                                        segment = th_delta[yu1:yu2,xu1:xu2]
                                        sa, segment_contours, segment_hierarchy = cv2.findContours(segment, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                        if len(segment_contours) > 0:
                                            c = max(segment_contours, key = cv2.contourArea)
                                            if cv2.contourArea(c) > (yu2-yu1)*(xu2-xu1)*0.6:
                                                #cv2.rectangle(new_frame_clr , (xu1, yu1), (xu2, yu2), (255, 255, 255), 2)
                                                vehicles.append(vehicle(len(vehicles)+tracked_vehicle_counter+2, xu1, yu1, xu2-xu1, yu2-yu1, 0, [None]*number_of_speeds, [None]*number_of_speeds, 0, True, i, 1, i,xu1,yu1,[],[]))
                                        
                                        segment = th_delta[yl1:yl2,xl1:xl2]
                                        sa, segment_contours, segment_hierarchy = cv2.findContours(segment, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                        if len(segment_contours) > 0:
                                            c = max(segment_contours, key = cv2.contourArea)
                                            if cv2.contourArea(c) > (yl2-yl1)*(xl2-xl1)*0.7:
                                                #cv2.rectangle(new_frame_clr , (xl1, yl1), (xl2, yl2), (255, 255, 255), 2)
                                                vehicles.append(vehicle(len(vehicles)+tracked_vehicle_counter+2, xl1, yl1, xl2-xl1, yl2-yl1, 0, [None]*number_of_speeds, [None]*number_of_speeds, 0, True, i, 1, i,xl1,yl1,[],[]))

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

                if centroids.size != 0 and new_centroids.size != 0:
                    # Calculate distance between all centroids and add to D array
                    D = dist.cdist(centroids, new_centroids)
                    # Loop through each row of D array and find minimum distance
                    for d_row in range(D.shape[0]):
                        # Filter out min distances over threshold value
                        maxDistance =  pixel_to_meter*10
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
                                matched_centroid_ids.append([src_vid,dst_vid])
                                matched_centroid_index.append([src_vindex,dst_vindex])
                        

                    # Set new vehicle counter to 0 (vehicles that have not been seen in previous frame)
                    new_vehicle_counter = 0
                    vehicles_to_remove = []

                    # Loop through each matched pair of centroids
                    for m in range(len(matched_centroid_ids)):
                        # Update vehicle speed history array (10 previous speed values held for averaging)
                        speed_counter = 1
                        for speed_index in range(number_of_speeds-1):
                            if vehicles_previous[matched_centroid_index[m][0]].speedx[speed_index] != None: 
                                speed_counter = speed_counter +1
                                # Move each speed back one frame in array
                                vehicles[matched_centroid_index[m][1]].speedx[speed_index+1] = vehicles_previous[matched_centroid_index[m][0]].speedx[speed_index]
                                vehicles[matched_centroid_index[m][1]].speedy[speed_index+1] = vehicles_previous[matched_centroid_index[m][0]].speedy[speed_index]
                        
                        # Update x, y instantaneous speeds
                        vehicles[matched_centroid_index[m][1]].speedx[0] = ((vehicles[matched_centroid_index[m][1]].x - vehicles_previous[matched_centroid_index[m][0]].x)/pixel_to_meter)*frame_rate
                        vehicles[matched_centroid_index[m][1]].speedy[0] = ((vehicles[matched_centroid_index[m][1]].y - vehicles_previous[matched_centroid_index[m][0]].y)/pixel_to_meter)*frame_rate
                        # Update 2D averaged speed
                        vehicles[matched_centroid_index[m][1]].speed = math.sqrt((sum(vehicles[matched_centroid_index[m][1]].speedx[0:speed_counter-1])/speed_counter)**2+(sum(vehicles[matched_centroid_index[m][1]].speedy[0:speed_counter-1])/speed_counter)**2)
                        
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
                    lost_time = frame_rate*5
                    legit_time = frame_rate*2
                    for v in tracked_vehicles:
                        if i-(lost_time) < v.lastFrame < i and v.visible_count > legit_time + 6 and new_frame.shape[1]*0.05 < v.x < new_frame.shape[1]*0.95:
                            if v.visible == True and v.id == 19:
                                print('Vehicle 19 lost.')                    
                            # Update position
                            speed_counter = 1
                            for speed_index in range(number_of_speeds-1):
                                if v.speedx[speed_index] != None: 
                                    speed_counter = speed_counter + 1
                            # Estimate change in x and y position
                            delta_x = int(((sum(v.speedx[3:speed_counter-1])/(speed_counter-3))/frame_rate)*pixel_to_meter)
                            delta_y = int(((sum(v.speedy[3:speed_counter-1])/(speed_counter-3))/frame_rate)*pixel_to_meter)
                            # Use average w and h
                            w_avg =  int(sum([v.size_history[sh_index][0] for sh_index in range(frame_rate*1,len(v.size_history)-5)])/(len(v.size_history)-frame_rate*1-5))
                            h_avg =  int(sum([v.size_history[sh_index][1] for sh_index in range(frame_rate*1,len(v.size_history)-5)])/(len(v.size_history)-frame_rate*1-5))
                            # Update x and y position
                            v.x = v.x + delta_x + int((v.w - w_avg)/2)
                            v.y = v.y + delta_y + int((v.h - h_avg)/2)
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
                
                # Loop through all vehicles in frame and draw a bounding box. If vehicle is not visible(position being estimated) draw it in different colour
                for v in vehicles:
                    if v.visible == True:
                        th_delta = draw_bb(th_delta, v.x, v.y, v.w, v.h, (100), 2, 'Car:' + str(v.id) + ' ' + str(round(v.speed*mps_mph,1)) + ' mph', 0, 0.3, (0, 0, 0))
                        new_frame_clr = draw_bb(new_frame_clr, v.x, v.y, v.w, v.h, (150, 255, 200), 2, 'Car:' + str(v.id) + ' ' + str(v.w) + ' ' + str(v.h), 0, 0.3, (0, 0, 0))
                    else:
                        new_frame_clr = draw_bb(new_frame_clr, v.x, v.y, v.w, v.h, (255, 255, 200), 2, 'Car:' + str(v.id) + ' ' + str(v.w) + ' ' + str(v.h), 0, 0.3, (0, 0, 0))
                        th_delta = draw_bb(th_delta, v.x, v.y, v.w, v.h, (175), 2, 'Car:' + str(v.id) + ' ' + str(round(v.speed*mps_mph,1)) + ' mph', 0, 0.3, (0, 0, 0))

                # Set vehicles_previous to current frame vehicles
                vehicles_previous = vehicles

                # Display frame 
                if show == 1:
                    cv2.imshow('1', diff)
                if show == 2:
                    cv2.imshow('2', thresh)
                if show == 3:
                    cv2.imshow('3', th_delta)
                if show == 4:
                    cv2.imshow('4', new_frame_clr)
                dim =  (int(new_frame.shape[1]/2), int(new_frame.shape[0]/2))
                if show == 5:
                    cv2.imshow('1', cv2.resize(diff, dim))
                    cv2.imshow('2', cv2.resize(thresh, dim))
                    cv2.imshow('3', cv2.resize(th_delta, dim))
                    cv2.imshow('4', cv2.resize(new_frame_clr, dim))
                numpy_horizontal_concat1 = np.concatenate((cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR), cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)), axis=1)
                numpy_horizontal_concat2 = np.concatenate((cv2.cvtColor(th_delta, cv2.COLOR_GRAY2BGR), new_frame_clr), axis=1)
                numpy_vertical_concat = np.concatenate((numpy_horizontal_concat1, numpy_horizontal_concat2), axis=0)
                if show == 6:
                    cv2.imshow('1', cv2.resize(numpy_vertical_concat, (int(new_frame.shape[1]), int(new_frame.shape[0]))))

                imgNumber = str(i)
                for d in range(5-len(imgNumber)):
                    imgNumber = '0' + imgNumber
                cv2.imwrite(detectionsFolder + r'\frames\\' + videoName + '_detections_' + imgNumber + '.jpg', new_frame_clr)
                cv2.waitKey(1)
    
    # Export tracked vehichles array as .obj file
    file_pi = open(os.path.join(trajectoriesFolder, videoName + '.obj'), 'wb')
    pickle.dump(tracked_vehicles, file_pi)
    # Return tracket vehicles array as result of function
    return tracked_vehicles
