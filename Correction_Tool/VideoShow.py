from threading import Thread
import cv2

class VideoShow:
    """
    Class that continuously shows a frame using a dedicated thread.
    """

    def __init__(self, frame=None):
        self.frame = frame
        self.stopped = False
        self.play = True
        self.rewind = False
        self.refPt = []
        self.cropping = False
        self.insert = False

    def click_and_crop(self,event, x, y, flags, param):
        # grab references to the global variables
    	global refPt, cropping
    	# if the left mouse button was clicked, record the starting
    	# (x, y) coordinates and indicate that cropping is being
    	# performed
    	if event == cv2.EVENT_LBUTTONDOWN:
    		self.refPt = [(x, y)]
    		self.cropping = True
    	# check to see if the left mouse button was released
    	elif event == cv2.EVENT_LBUTTONUP:
    		# record the ending (x, y) coordinates and indicate that
    		# the cropping operation is finished
    		self.refPt.append((x, y))
    		self.cropping = False
    		# draw a rectangle around the region of interest
    		cv2.rectangle(self.frame, self.refPt[0], self.refPt[1], (0, 255, 0), 2)

    def insertObject(self):
        self.refPt = []
        clone = self.frame.copy()
        cv2.namedWindow("image",cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('image', self.click_and_crop)
        while True:
            # display the image and wait for a keypress

            cv2.imshow("image",self.frame)
            key = cv2.waitKey(1) & 0xFF
            # if the 's' key is pressed, reset the cropping region
            if key == ord("s"):
                self.frame = clone.copy()
            # if the 'c' key is pressed, break from the loop
            elif key == ord("c"):
                break

    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def show(self):
        while not self.stopped:
            if self.play == True:

                cv2.namedWindow("Video",cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Video", 570,348)
                cv2.imshow("Video", self.frame)

                #cv2.waitKey(1)

                if cv2.waitKey(1) & 0xFF == ord("q"): # end
                    self.stopped = True
            elif self.insert == True:
                self.insertObject()
                self.insert = False

            if cv2.waitKey(1) & 0xFF == ord('p'): # play/pause Video
                self.play = not self.play

                if self.play == True:
                    print("Playing")
                else:
                    print("Paused")

            if cv2.waitKey(1) & 0xFF == ord('r'): # rewind video
                self.rewind = not self.rewind

                if self.rewind == True:
                    print("Rewinding")
                else:
                    print("Playing")

    def stop(self):
        self.stopped = True


    def addFrame(self, frame):
        self.frame = frame
