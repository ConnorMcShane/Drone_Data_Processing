import csv
import argparse
import cv2
import os
import numpy as np
from VideoShow import VideoShow

def deleteErroneousObjects(frame, data,v_id, changes, changes_csv):
    '''
    function removes the vehicle ID selected by the user. This removes the information
    regarding the bboxes for that vehicle ID.

    returns the lists for the updated csv file and the change log csv file
    '''

    print("Vehicle IDs in this frame:")
    for i in range(len(v_id)):
        print(v_id[i])

    delObj = input("Enter vehicle ID you would like to remove: ")

    if delObj in v_id:
        count = 0
        orig_len = len(data)
        for row in range(1,len(data)):

            if orig_len-count == row:
                break
            else:
                if data[row][1] == delObj:
                    change = len(changes)+1
                    changes.append([change,data[row][0], data[row][1],'Removed',data[row][2],data[row][3],data[row][4],data[row][5]])
                    changes_csv.writerow([change,data[row][0], data[row][1],'Removed',data[row][2],data[row][3],data[row][4],data[row][5]])
                    del data[row]
                    count += 1

    return data, changes

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False
def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True
	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		cropping = False
		# draw a rectangle around the region of interest
		#cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)


def insertNewObjects(frame, data, cap, timeStamp, row,changes_csv,changes, video_shower):
    '''
    function to allow user to draw missed bounding boxes in frame.
    Tracks new objects until out of frame and adds bbox information to the original csv
    file and the change log csv file

    Returns the lists for the updated csv file and the change log csv file
    '''

    # set bounding box as green
    colours = (0,255,0)


    '''
    allows user to select new bbox in frame

    Press 'Esc' to when done
    '''
    print("Draw ROI from top left to bottom right with mouse. Press c when done.")

    while len(video_shower.refPt) != 2:
        pass

    print("Loop broken")
    bbox = [video_shower.refPt[0][0],video_shower.refPt[0][1], abs(video_shower.refPt[1][0]-video_shower.refPt[0][0]),abs(video_shower.refPt[1][1]-video_shower.refPt[0][1])]
    # sets the tracker type
    trackerType = "CSRT"

    # create tracker object
    tracker = cv2.TrackerCSRT_create()

    '''
    Adds the vehicle ID for the new vehicle

    Adds ROI to be tracked through following frames
    '''
    vehicle_id = int(max(l[1] for l in data[1:]))+1
    print(vehicle_id)
    ok = tracker.init(frame, bbox)

    '''
    loop reads new frames and updates location of tracked Objects

    adds them to csv files with points in metres

    breaks loop when object no longer in frame
    '''
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if row < len(data):
            timestamp = data[row][0]
        else:
            break

        success, boxes = tracker.update(frame)

        if not success:
            break

        '''
        loops through tracked objects and updates information csv file and adds to change log
        '''

        x_centre = int(boxes[0] + (int(boxes[2])/2))*0.024
        y_centre = int(boxes[1] + (int(boxes[3])/2))*0.024
        length = int(boxes[2])*0.024
        width = int(boxes[3])*0.024
        data.append([timeStamp,vehicle_id,x_centre,y_centre,length,width])

        change = len(changes)+1
        changes.append([change,timeStamp,vehicle_id,'Added',x_centre,y_centre,length,width])
        changes_csv.writerow([change,timeStamp,vehicle_id,'Added',x_centre,y_centre,length,width])


        while data[row][0] == timestamp:
            row += 1

    # sorts csv data while preserving headers
    data[1:] = sorted(data[1:], key=lambda x: x[0])

    # returns csv data
    return data, changes

def threadVideoShow(source=0):
    """
    Dedicated thread for showing video frames with VideoShow object.
    Main thread grabs video frames.

    Adds bounding boxes from initial csv file to frames and includes rewind functionality
    for the video stream

    Allows user to edit bbox detections when video is Paused

    Press 'p' to pause/play and 'r' to rewind and continue
    """

    cap = cv2.VideoCapture(source)

    (grabbed, frame) = cap.read()

    video_shower = VideoShow(frame).start()
    with open('Change Log.csv','a') as ChangeLog:
        changes = []
        changes_csv = csv.writer(ChangeLog, delimiter=',')

        if os.path.getsize('Change Log.csv') == 0:
            changes_csv.writerow(['Change','Timestamp (s)','Vehicle ID', 'Type of Change', 'x_centre (m)', 'y_centre (m)', 'length (m)','width (m)'])

        row = 1
        with open('TrackedUnfiltered-100120-mav1-02-1403 - Sheet1.csv','r') as InitialCSV:
            data = list(csv.reader(InitialCSV))


        counter = 1
        while True:

            if video_shower.play == True:
                v_id = []
                if row < len(data):
                    timestamp = data[row][0]
                else:
                    break

                (grabbed, frame) = cap.read()
                if not grabbed or video_shower.stopped:
                    video_shower.stop()
                    break

                '''
                draws bboxes to frame and iterates through the rows to next timestep

                pauses video if rewound to 1st frame
                '''
                while data[row][0] == timestamp:
                    cv2.rectangle(frame,(int(float(data[row][2])/0.024)-int((float(data[row][4])/0.024)/2),int(float(data[row][3])/0.024)-int((float(data[row][5])/0.024)/2)),(int(float(data[row][2])/0.024)+int((float(data[row][4])/0.024)/2),int(float(data[row][3])/0.024)+int((float(data[row][5])/0.024)/2)),(0,255,0),3)
                    v_id.append(data[row][1])
                    if row < len(data):
                        if video_shower.rewind == False:
                            row += 1
                        else:
                            row -= 1
                            if row < 1:
                                row = 1

                                video_shower.rewind = False
                                video_shower.play = False
                        if row == len(data):
                            break

                video_shower.addFrame(frame)

                '''
                determines the direction of the video based on the rewind flag
                '''
                if video_shower.rewind == False:
                    counter += 1
                else:
                    counter -= 1
                    if data[row][0] != data[row-1][0]:
                        row -= 1

                    if counter < 0:
                        counter = 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, counter)

                '''
            Runs when video is paused
                '''
            else:
                #video_shower.stop()
                userInput = input("Insert/Remove/Nothing: ")

                if userInput == "Insert":
                    #video_shower.stop()
                    #cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
                    #cv2.imshow("image", frame)
                    #cv2.waitKey(0)
                    # bbox editing tool
                    video_shower.insert = True
                    data, changes = insertNewObjects(frame, data, cap, timestamp, row, changes_csv, changes,video_shower)
                    #video_shower.start()
                elif userInput == "Remove":
                    data, changes = deleteErroneousObjects(frame, data, v_id, changes, changes_csv)
                elif userInput == "Nothing, playing":
                    video_shower.play = True
                    video_shower.start()
                    continue
                else:
                    print("Re-enter choice!")
                    continue

                '''
                Updates the CSV file with the updated bbox information
                '''
                with open("TrackedUnfiltered-100120-mav1-02-1403 - Sheet1.csv", 'w') as InitialCSV:
                    Update_CSV = csv.writer(InitialCSV, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    headings = data[0]
                    Update_CSV.writerow(headings)

                    for i in range(1,len(data)):
                        Update_CSV.writerow(data[i])

                #video_shower.start()



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", "-s", default=0,
        help="Path to video file or integer representing webcam index"
            + " (default 0).")

    args = vars(ap.parse_args())

    threadVideoShow(args["source"])

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
