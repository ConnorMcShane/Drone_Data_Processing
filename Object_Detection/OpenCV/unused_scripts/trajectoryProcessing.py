import numpy as np
import pickle
import os
import cv2
import math

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
        self.visible_count = visible_count
        self.startFrame = startFrame
        self.startX = startX
        self.startY = startY
        self.trajectory = trajectory
        self.size_history = size_history
    
def trajectoryProcessing(rootFolder, videoName, fps, numberOfFrames):
    stabilizedFolder = rootFolder + r'\stabilized\\' + videoName + r'\\'
    backgroundFolder = rootFolder + r'\background\\' + videoName + r'\\'
    detectionsFolder = rootFolder + r'\detections\\' + videoName + r'\\'
    originalFolder = rootFolder + r'\original\\'
    warpedFolder = rootFolder + r'\warped\\' + videoName + r'\\'
    trajectoriesFolder = rootFolder +  r'\trajectories\\' + videoName + r'\\'

    folders = [rootFolder, rootFolder + r'\background\\', rootFolder + r'\stabilized\\', stabilizedFolder, rootFolder + r'\detections\\', rootFolder +  r'\trajectories\\', backgroundFolder, detectionsFolder, detectionsFolder + r'\frames\\', detectionsFolder + r'\videos\\', originalFolder, trajectoriesFolder,  rootFolder + r'\warped\\', warpedFolder, warpedFolder + r'\frames\\' , warpedFolder + r'\videos\\']

    video = warpedFolder + r'\videos\\' + videoName + r'.avi'

    for folder in folders:
        if not os.path.isdir(folder):
            os.mkdir(folder)

    
    file_pi_path = trajectoriesFolder + videoName + r'.obj'
    file_pi = open(file_pi_path, 'rb')
    tracked_vehicles = pickle.load(file_pi)
    framerate = fps
    pixel_to_meter = 50/7.5
    mps_mph = 2.237

    def predict(x, y, speedx, speedy):
        x = x + speedx
        y = y + speedy
        return x, y
        
    # Function to draw bounding boxes
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

    # Function to draw bounding boxes rotated
    def draw_bb_r(img, x, y, w, h, theta, colour, thickness, text, font, size, text_colour, x1, y1, x2, y2):
        cv2.rectangle(img , (x, y), (x+w, y+h), colour, thickness)
        cv2.rectangle(img , (x, y-8), (x+int(len(text)*5.8), y), colour, -1)
        cv2.putText(img, text, (x, y-1), font, size, text_colour, 1, cv2.LINE_AA)
        #cv2.line(frame, (x1, y1), (x2, y2), colour, 2)
        '''rotatedRect = (x1, y1),(w,h),theta
        box = cv2.boxPoints(rotatedRect) 
        box = np.int0(box)
        cv2.drawContours(img,[box],0,colour,2)'''
        return img

    # Function to calculate rolling average
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

    cap = cv2.VideoCapture(video)
    ret, frame = cap.read()

    rollingAverageSampleSize = 15

    tracked_vehicles_temp = []

    for v in range(len(tracked_vehicles)):
        if tracked_vehicles[v].visible_count > 29*2:
            tracked_vehicles_temp.append(tracked_vehicles[v])

    tracked_vehicles = tracked_vehicles_temp

    x = [[(tracked_vehicles[v_index].startX+tracked_vehicles[v_index].trajectory[x_index][0]) for x_index in range(len(tracked_vehicles[v_index].trajectory))] for v_index in range(len(tracked_vehicles))]
    y = [[(tracked_vehicles[v_index].startY+tracked_vehicles[v_index].trajectory[y_index][1]) for y_index in range(len(tracked_vehicles[v_index].trajectory))] for v_index in range(len(tracked_vehicles))]
    w = [[(tracked_vehicles[v_index].size_history[w_index][0]) for w_index in range(len(tracked_vehicles[v_index].trajectory))] for v_index in range(len(tracked_vehicles))]
    h = [[(tracked_vehicles[v_index].size_history[h_index][1]) for h_index in range(len(tracked_vehicles[v_index].trajectory))] for v_index in range(len(tracked_vehicles))]
    print(h)
    #w_avg = [int(sum([(tracked_vehicles[v_index].size_history[w_index][0]) for w_index in range(len(tracked_vehicles[v_index].trajectory))])/len(tracked_vehicles[v_index].trajectory)) for v_index in range(len(tracked_vehicles))]
    #w_avg = [int(sum([(w[v_index][w_index]) for w_index in range((int(len(w[v_index])/2))-5,(int(len(w[v_index])/2))+5)])/10) for v_index in range(len(w))]
    w_avg = [max(set(w[v_index]), key=w[v_index].count) for v_index in range(len(w))]
    #h_avg = [int(sum([(tracked_vehicles[v_index].size_history[h_index][1]) for h_index in range(len(tracked_vehicles[v_index].trajectory))])/len(tracked_vehicles[v_index].trajectory)) for v_index in range(len(tracked_vehicles))]
    #h_avg = [int(sum([(h[v_index][h_index]) for h_index in range((int(len(h[v_index])/2))-5,(int(len(h[v_index])/2))+5)])/10) for v_index in range(len(h))]
    h_avg = [max(set(h[v_index]), key=h[v_index].count) for v_index in range(len(h))]
    x_ravg = [[int(rollingAverage(x[v_index], x_index, rollingAverageSampleSize)) for x_index in range(len(x[v_index]))] for v_index in range(len(x))]
    y_ravg = [[int(rollingAverage(y[v_index], y_index, rollingAverageSampleSize)) for y_index in range(len(y[v_index]))] for v_index in range(len(y))]
    x_kf = x
    y_kf = y

    #Kalman filtering
    speed_num = 29
    maxspeed = ((120/mps_mph)/framerate)*pixel_to_meter
    for v in range(len(tracked_vehicles)):
        for ts in range(speed_num+1, len(x_kf[v])):
            speedx = (x_kf[v][ts-2]-x_kf[v][ts-speed_num-2])/speed_num
            speedy = (y_kf[v][ts-2]-y_kf[v][ts-speed_num-2])/speed_num

            x_predict, y_predict = predict(x_kf[v][ts-1], y_kf[v][ts-1], speedx, speedy)
            if abs(speedx) > 0:
                current_speed = abs((x_kf[v][ts] - x_kf[v][ts-1]))
                last_speed = abs(speedx)
                if current_speed > maxspeed:
                    relative_speed_ratio = 0
                else:
                    relative_speed_ratio = current_speed/maxspeed
                if current_speed > last_speed:
                    measurement_confidencex = 1-((abs(current_speed-last_speed)/current_speed))#*(relative_speed_ratio**2))
                else:
                    measurement_confidencex = 1-((abs(last_speed-current_speed)/last_speed))#*(relative_speed_ratio**2))
            else:
                measurement_confidencex = 1
            if abs(speedy) > 0:
                current_speed = abs((y_kf[v][ts] - y_kf[v][ts-1]))
                last_speed = abs(speedy)
                if current_speed > maxspeed:
                    relative_speed_ratio = 0
                else:
                    relative_speed_ratio = current_speed/maxspeed
                if current_speed > last_speed:
                    measurement_confidencey = 1-((abs(current_speed-last_speed)/current_speed))#*(relative_speed_ratio**2))
                else:
                    measurement_confidencey = 1-((abs(last_speed-current_speed)/last_speed))#*(relative_speed_ratio**2))
            else:
                measurement_confidencey = 1
            measurement_confidence = ((measurement_confidencex+measurement_confidencey)/4 + 0.5)
            
            print(measurement_confidence)
            x_kf[v][ts] = int(x_kf[v][ts]*measurement_confidence + x_predict*(1-measurement_confidence))
            y_kf[v][ts] = int(y_kf[v][ts]*measurement_confidence + y_predict*(1-measurement_confidence))



    #x_ravg = x_kf
    #y_ravg = y_kf

    # Edge correction
    appearance_point = []
    disappearance_point = []

    for v in range(len(tracked_vehicles)):
        appearance_point.append(0)
        disappearance_point.append(0)
        for i in range(len(x[v])):
            if x[v][i] < (frame.shape[1]-w_avg[v]) and x[v][i] > w_avg[v] and y[v][i] < (frame.shape[0]-h_avg[v]) and y[v][i] > h_avg[v]:
                    if appearance_point[v] == 0:
                        appearance_point[v] = i
            if x[v][i] > (frame.shape[1]-w_avg[v]*3) or x[v][i] < w_avg[v]*2 or y[v][i] > (frame.shape[0]-h_avg[v]*3) or y[v][i] < h_avg[v]*2:
                    if disappearance_point[v] == 0 and i-appearance_point[v] > framerate*3:
                        if appearance_point[v] != 0:
                            disappearance_point[v] = i


    for v in range(len(tracked_vehicles)):
        if appearance_point[v] > 1:
            x_speed = (x_ravg[v][appearance_point[v]] - x_ravg[v][(appearance_point[v]+(framerate*1))])/(framerate*1)
            y_speed = (y_ravg[v][appearance_point[v]] - y_ravg[v][(appearance_point[v]+(framerate*1))])/(framerate*1)

            for hx in range(len(x_ravg[v][0:appearance_point[v]])):
                x_ravg[v][hx] = int(x_ravg[v][appearance_point[v]] + x_speed*(len(x_ravg[v][0:appearance_point[v]]) - hx))
            for hy in range(len(y_ravg[v][0:appearance_point[v]])):
                y_ravg[v][hy] = int(y_ravg[v][appearance_point[v]] + y_speed*(len(y_ravg[v][0:appearance_point[v]]) - hy))
        if disappearance_point[v] > 1:
            x_speed = (x_ravg[v][disappearance_point[v]] - x_ravg[v][(disappearance_point[v]-(framerate*2))])/(framerate*2)
            y_speed = (y_ravg[v][disappearance_point[v]] - y_ravg[v][(disappearance_point[v]-(framerate*2))])/(framerate*2)

            for hx in range(len(x_ravg[v][0:disappearance_point[v]]), len(x_ravg[v])):
                x_ravg[v][hx] = int(x_ravg[v][disappearance_point[v]] - x_speed*(len(x_ravg[v][0:disappearance_point[v]]) - hx))
            for hy in range(len(y_ravg[v][0:disappearance_point[v]]), len(y_ravg[v])):
                y_ravg[v][hy] = int(y_ravg[v][disappearance_point[v]] - y_speed*(len(y_ravg[v][0:disappearance_point[v]]) - hy))

    # Get speeds

    x_speed = [[(x_ravg[v_index][x_index]-x_ravg[v_index][x_index-1]) for x_index in range(len(x_ravg[v_index]))] for v_index in range(len(x_ravg))]
    y_speed = [[(y_ravg[v_index][y_index]-y_ravg[v_index][y_index-1]) for y_index in range(len(y_ravg[v_index]))] for v_index in range(len(y_ravg))]

    # Get angles
    angles = []
    for v in range(len(tracked_vehicles)):
        angles.append([])
        delta_x = []
        delta_x = [(x_ravg[v][i+2] - x_ravg[v][i]) for i in range(len(x_ravg[v]) - 2)]
        delta_x.insert(0, x_ravg[v][2] - x_ravg[v][0])
        delta_x.append(x_ravg[v][len(x_ravg[v])-1] - x_ravg[v][len(x_ravg[v])-3])

        delta_y = []
        delta_y = [(y_ravg[v][i+2] - y_ravg[v][i]) for i in range(len(y_ravg[v]) - 2)]
        delta_y.insert(0, y_ravg[v][2] - y_ravg[v][0])
        delta_y.append(y_ravg[v][len(y_ravg[v])-1] - y_ravg[v][len(y_ravg[v])-3])

        delta_x = [rollingAverage(delta_x, x_index, 15) for x_index in range(len(delta_x))]
        delta_y = [rollingAverage(delta_y, y_index, 15) for y_index in range(len(delta_y))]

        for i in range(len(delta_x)):
            delta_x_i = delta_x[i]
            delta_y_i = delta_y[i]
            if delta_x_i == 0:
                theta = 0
            else:
                theta = math.atan(delta_y_i/delta_x_i) # In radians!
            angles[v].append(theta)
            # Print(theta)

    angles = [[round(angles[v_index][t_index],4) for t_index in range(len(angles[v_index]))] for v_index in range(len(angles))]

    # Draw boxes
    length = 50
    for j in range(int(numberOfFrames)):
        cap.set(1, j)
        ret, frame = cap.read()
        for v in range(len(tracked_vehicles)):
            sf = tracked_vehicles[v].startFrame
            sx = tracked_vehicles[v].startX
            sy = tracked_vehicles[v].startY
            if tracked_vehicles[v].startFrame <= j:
                if tracked_vehicles[v].startFrame + len(tracked_vehicles[v].trajectory) > j:
                    if appearance_point[v] > 0 and disappearance_point[v] > 0:
                        #frame = draw_bb(frame, x_ravg[v][j-sf], y_ravg[v][j-sf], w_avg[v], h_avg[v], (150, 255, 255), 2, 'theta = ' + str(round(theta_avg[v][j-sf], 2)), 0, 0.3, (0, 0, 0))
                        if x_speed[v][j-sf] != 0:
                            x_sign = x_speed[v][j-sf]/abs(x_speed[v][j-sf])
                        if y_speed[v][j-sf] != 0:
                            y_sign = y_speed[v][j-sf]/abs(y_speed[v][j-sf])
                        x1 = int(x_ravg[v][j-sf] + (w_avg[v])/2)
                        y1 = int(y_ravg[v][j-sf] + h_avg[v]*0.5)
                        x2 = int(x1 + x_sign*w_avg[v]*(math.cos(angles[v][j-sf]))/2)
                        y2 = int(y1 - w_avg[v]*(math.sin(angles[v][j-sf])))
                        frame = draw_bb_r(frame, x_ravg[v][j-sf], y_ravg[v][j-sf], w_avg[v], h_avg[v], angles[v][j-sf],(255, 250, 200), 2, 'V' + str(v), 0, 0.3, (0, 0, 0), x1, y1, x2, y2)
                    else:
                        #frame = draw_bb(frame, x_ravg[v][j-sf], y_ravg[v][j-sf], w_avg[v], h_avg[v], (255, 100, 100), 2, 'theta = ' + str(round(theta_avg[v][j-sf], 2)), 0, 0.3, (0, 0, 0))
                        if x_speed[v][j-sf] != 0:
                            x_sign = x_speed[v][j-sf]/abs(x_speed[v][j-sf])
                        if y_speed[v][j-sf] != 0:
                            y_sign = y_speed[v][j-sf]/abs(y_speed[v][j-sf])
                        x1 = int(x_ravg[v][j-sf] + (w_avg[v])/2)
                        y1 = int(y_ravg[v][j-sf] + h_avg[v]*0.5)
                        x2 = int(x1 + x_sign*w_avg[v]*(math.cos(angles[v][j-sf]))/2)
                        y2 = int(y1 - w_avg[v]*(math.sin(angles[v][j-sf])))
                        frame = draw_bb_r(frame, x_ravg[v][j-sf], y_ravg[v][j-sf], w_avg[v], h_avg[v], angles[v][j-sf],(150, 150, 255), 2, 'V' + str(v), 0, 0.3, (0, 0, 0), x1, y1, x2, y2)
        
        cv2.imshow('trajectoryPerspective', frame)
        cv2.waitKey(20)
        print(j)
