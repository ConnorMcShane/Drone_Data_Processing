import numpy as np
import matplotlib.pyplot as plt
from math import *
import cv2

def filterTajectories(tracked_vehicles, numberOfFrames):

    x = [[(tracked_vehicles[v_index].startX+tracked_vehicles[v_index].trajectory[x_index][0]) for x_index in range(len(tracked_vehicles[v_index].trajectory))] for v_index in range(len(tracked_vehicles))]
    y = [[(tracked_vehicles[v_index].startY+tracked_vehicles[v_index].trajectory[y_index][1]) for y_index in range(len(tracked_vehicles[v_index].trajectory))] for v_index in range(len(tracked_vehicles))]
    w = [[(tracked_vehicles[v_index].startX+tracked_vehicles[v_index].size_history[w_index][0]) for w_index in range(len(tracked_vehicles[v_index].trajectory))] for v_index in range(len(tracked_vehicles))]
    h = [[(tracked_vehicles[v_index].startY+tracked_vehicles[v_index].size_history[h_index][1]) for h_index in range(len(tracked_vehicles[v_index].trajectory))] for v_index in range(len(tracked_vehicles))]
    X = x
    Y = y


    #fig = plt.figure()
    #ax = fig.add_subplot(111, aspect='equal')
    #for v in range(len(tracked_vehicles)):
    #    ax.plot(x[v], y[v], color='blue', marker='o', linestyle='dashed', linewidth=1, markersize=2)
    #plt.axis([0,1728,400,800])
    #plt.show()
    
    '''# gaussian function
    def f(mu, sigma2, x):
        coefficient = 1.0/sqrt(2.0*pi*sigma2)
        exponential = exp(-0.5*(x-mu)**2/sigma2)
        return coefficient*exponential
    
    # the update function
    def update(mean1, var1, mean2, var2):
        new_mean = (var2*mean1 + var1*mean2)/(var2+var1)
        new_var = 1/(1/var2 + 1/var1)

        return [new_mean, new_var]
    
    # the predict function
    def predict(mean1, var1, mean2, var2):
        new_mean = mean1 + mean2
        new_var = var1 + var2

        return [new_mean, new_var]

    def rollingAverage(array, position, numberOfPoints):
        rollingAverage = int(sum(array[int(position - (numberOfPoints/2) - 0.5):int(position + (numberOfPoints/2) - 0.5)])/numberOfPoints)
        return rollingAverage
        
    for n in range(len(measurements)):
        # measurement update, with uncertainty
        mu, sig = update(mu, sig, measurements[n], measurement_sig)
        print('Update: [{}, {}]'.format(mu, sig))

        # motion update, with uncertainty
        mu, sig = predict(mu, sig, motions[n], motion_sig)
        print('Predict: [{}, {}]'.format(mu, sig))

    

    for v in range(len(x)):
        # initial parameters
        measurement_sig = 400.
        motion_sig = 200.
        mu = x[v][0]
        sig = 1000.
        for n in range(len(x[v])-1):
            # measurement update, with uncertainty
            mu, sig = update(mu, sig, x[v][n], measurement_sig)
            #print('Update: [{}, {}]'.format(mu, sig))
            X[v][n] = int(mu)

            # motion update, with uncertainty
            mu, sig = predict(mu, sig, x[v][n+1]-x[v][n], motion_sig)
            #print('Predict: [{}, {}]'.format(mu, sig))
            #X[v][n] = int(mu)

    for v in range(len(y)):
        # initial parameters
        measurement_sig = 400.
        motion_sig = 200.
        mu = y[v][0]
        sig = 1000.
        for n in range(len(y[v])-1):
            # measurement update, with uncertainty
            mu, sig = update(mu, sig, y[v][n], measurement_sig)
            #print('Update: [{}, {}]'.format(mu, sig))
            Y[v][n] = int(mu)

            # motion update, with uncertainty
            mu, sig = predict(mu, sig, y[v][n+1]-y[v][n], motion_sig)
            #print('Predict: [{}, {}]'.format(mu, sig))
            #Y[v][n] = int(mu)'''
    
    def rollingAverage(array, position, numberOfPoints):
        if numberOfPoints/2 > position:
            numberOfPoints = int(position*2)+1
            print('Lower end correction. Number of points = ' + str(numberOfPoints))
        if numberOfPoints/2 > len(array)-position:
            numberOfPoints = int((len(array)-position)*2)-1
            print('Upper end correction. Number of points = ' + str(numberOfPoints))
        rollingAverage = 0
        for i in range(numberOfPoints):
            rollingAverage = rollingAverage + array[position - int(numberOfPoints/2 - 0.5) + i]
        rollingAverage = int(rollingAverage/numberOfPoints)
        return rollingAverage
        
    def draw_bb(img, x, y, w, h, colour, thickness, text, font, size, text_colour):
        cv2.rectangle(img , (x, y), (x+w, y+h), colour, thickness)
        cv2.rectangle(img , (x, y-8), (x+60, y), colour, -1)
        cv2.putText(img, text, (x, y-1), font, size, text_colour, 1, cv2.LINE_AA)

    for v in range(len(y)):
        for n in range(len(y[v])):
            
            X[v][n] = rollingAverage(x[v], n, 13)
            Y[v][n] = rollingAverage(y[v], n, 13)
            print(str(x[v][n]) + ' to ' + str(X[v][n]))

    cap = cv2.VideoCapture(r'C:\Users\cmcshan1\Documents\DroneFootage\Drone_Videos\stabilized\videos\DJI_0059.avi')
    ret, frame = cap.read()

    numberOfFrames = numberOfFrames
    for j in range(int(numberOfFrames)):
        cap.set(1, j)
        ret, frame = cap.read()
        for v in range(len(tracked_vehicles)):
            if tracked_vehicles[v].visible_count > 29*5:
                if tracked_vehicles[v].startFrame <= j:
                    if tracked_vehicles[v].lastFrame > j:
                        #draw_bb(frame, int(tracked_vehicles[v].startX + tracked_vehicles[v].trajectory[j-tracked_vehicles[v].startFrame][0]), int(tracked_vehicles[v].startY + tracked_vehicles[v].trajectory[j-tracked_vehicles[v].startFrame][1]), tracked_vehicles[v].w, tracked_vehicles[v].h, (200, 255, 255), 2, 'C:' + str(tracked_vehicles[v].id), 0, 0.3, (0, 0, 0))
                        draw_bb(frame, X[v][j-tracked_vehicles[v].startFrame], Y[v][j-tracked_vehicles[v].startFrame], w[v][j-tracked_vehicles[v].startFrame], h[v][j-tracked_vehicles[v].startFrame], (255, 255, 200), 2, 'C:' + str(tracked_vehicles[v].id), 0, 0.3, (0, 0, 0))
        cv2.imshow('Background Subtracted', frame)
        cv2.waitKey(1)