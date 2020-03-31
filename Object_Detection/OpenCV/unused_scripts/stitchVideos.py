import numpy as np
import argparse
import cv2
import os
import math


video1Folder = r'D:\Connor\Autoplex\Data\Drone\Video\Warped\100120-mav2-01-1403\frames'
video2Folder = r'D:\Connor\Autoplex\Data\Drone\Video\Warped\100120-mav1-02-1403\frames'
maxFeatures = 20000
contrastThresh = 0.01

def stitch(image1, image2):
    sift = cv2.xfeatures2d.SIFT_create(nfeatures = maxFeatures, nOctaveLayers = 3, contrastThreshold = contrastThresh, edgeThreshold = 4, sigma = 1.6)

    cropRatio = 5

    h, w = image1.shape[0], image1.shape[1]
    image1cropped = image1[0:h, int(w-(w/cropRatio)):w]
    h, w = image2.shape[0], image2.shape[1]
    image2cropped = image2[0:h, 0:int(w/cropRatio)]

    kp1, desc1 = sift.detectAndCompute(image1cropped,None)
    kp2, desc2 = sift.detectAndCompute(image2cropped,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1,desc2,k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.6*n.distance:
            keyPoint1y = kp1[m.queryIdx].pt[1]
            keyPoint2y = kp2[m.trainIdx].pt[1]
            if math.sqrt((keyPoint1y-keyPoint2y)**2) < h/8:
                good.append(m)

    #img3 = cv2.drawMatches(image1cropped,kp1,image2cropped,kp2,good[:10], None, flags=2, matchColor=(0,255,0))
    #dim = (int(img3.shape[1]/2), int(img3.shape[0]/2))
    #cv2.imshow('matches',  cv2.resize(img3, dim))
    #cv2.waitKey(1)
    #print('done')

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    src_x = [src_pts[j][0][0] for j in range(len(src_pts))]
    src_y = [src_pts[j][0][1] for j in range(len(src_pts))]
    dst_x = [dst_pts[j][0][0] for j in range(len(dst_pts))]
    dst_y = [dst_pts[j][0][1] for j in range(len(dst_pts))]

    avg_src_x = int(sum(src_x)/len(src_x))
    avg_src_y = int(sum(src_y)/len(src_y))
    avg_dst_x = int(sum(dst_x)/len(dst_x))
    avg_dst_y = int(sum(dst_y)/len(dst_y))
    x_offset = image1cropped.shape[1] - avg_dst_x + avg_src_x

    src_line = np.polyfit(src_x, src_y, 1)
    dst_line = np.polyfit(dst_x, dst_y, 1)

    y1 = int(src_line[0]*0 + src_line[1])
    x1 = 0
    y2 = int(src_line[0]*image1cropped.shape[1] + src_line[1])
    x2 = image1cropped.shape[1]
    angle1 = np.degrees(math.atan(x2/(y2-y1)))
    image1cropped = cv2.line(image1cropped, (x1, y1), (x2, y2), (0,255,0), 2)
    image1cropped = cv2.circle(image1cropped, (avg_src_x, avg_src_y), 5, (0,0,255), 2) 

    y1 = int(dst_line[0]*0 + dst_line[1])
    x1 = 0
    y2 = int(dst_line[0]*image2cropped.shape[1] + dst_line[1])
    x2 = image2cropped.shape[1]
    angle2 = np.degrees(math.atan(x2/(y2-y1)))
    image2cropped = cv2.line(image2cropped, (x1, y1), (x2, y2), (0,255,0), 2)
    image2cropped = cv2.circle(image2cropped, (avg_dst_x, avg_dst_y), 5, (0,0,255), 2)

    angleDiff = angle1-angle2
    rot_mat = cv2.getRotationMatrix2D((avg_dst_x, avg_dst_y), angleDiff/2, 1.0)
    rotatedIMG = cv2.warpAffine(image2, rot_mat, image2.shape[1::-1], flags=cv2.INTER_LINEAR)

    dim = (int(rotatedIMG.shape[1]/2), int(rotatedIMG.shape[0]/2))
    cv2.imshow('rotatedIMG',  cv2.resize(rotatedIMG, dim))
    #dim = (int(image1cropped.shape[1]/2), int(image1cropped.shape[0]/2))
    #cv2.imshow('image1cropped',  cv2.resize(image1cropped, dim))
    #dim = (int(image2cropped.shape[1]/2), int(image2cropped.shape[0]/2))
    #cv2.imshow('image2cropped',  cv2.resize(image2cropped, dim))

    combinedIMG = np.zeros((h, w*2-x_offset, 3), dtype = "uint8")
    combinedIMG[0:h, int(w-x_offset):int(w*2)-x_offset] = rotatedIMG
    combinedIMG[0:h, 0:w] = image1
    cv2.imshow('warped', cv2.resize(combinedIMG, (int(combinedIMG.shape[1]/2), int(combinedIMG.shape[0]/2))))
    cv2.waitKey(1)

    print('done')
    #M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 1.0)
    #img3 = cv2.warpPerspective(image2, np.linalg.inv(M),(image2.shape[1], image2.shape[0]))
    #h, w, _ = img3.shape
    #img3_cropped = img3[int(h*0.01):int(h*0.99), int(w*0.01):int(w*0.99)]
    #img3_cropped = cv2.drawKeypoints(frame, new_kp, img3_cropped, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(255,255,255))
    
    #combinedIMG = np.zeros((h, w*2, 3), dtype = "uint8")
    #combinedIMG[0:h, int(w*0.75):int(w*1.75)] = img3
    #combinedIMG[0:h, 0:w] = image1
    #cv2.imshow('warped', cv2.resize(combinedIMG, (int(combinedIMG.shape[1]/2), int(combinedIMG.shape[0]/2))))
    #cv2.waitKey(1)
    
    #print('done')



filesVideo1 = []
for file in os.listdir(video1Folder):
    if file.endswith(r'.jpg'):
        filesVideo1.append(os.path.join(video1Folder, file))

filesVideo2 = []
for file in os.listdir(video2Folder):
    if file.endswith(r'.jpg'):
        filesVideo2.append(os.path.join(video2Folder, file))

for i in range(len(filesVideo2)):

    images = []
    #image = cv2.imread(filesVideo1[5])
    image = cv2.imread(r'D:\Connor\Autoplex\Data\Drone\Video\Test\1.jpg')
    images.append(image)
    #image = cv2.imread(filesVideo2[0])
    image = cv2.imread(r'D:\Connor\Autoplex\Data\Drone\Video\Test\2.jpg')
    images.append(image)

    stitch(images[0], images[1])

    stitcher = cv2.createStitcher()
    (status, stitched) = stitcher.stitch(images)

    if status == 0:
        dim = (int(stitched.shape[1]/4), int(stitched.shape[0]/4))
        cv2.imshow("Stitched", cv2.resize(stitched, dim))
        cv2.waitKey(0)
    else:
        print("[INFO] image stitching failed ({})".format(status))


