from Object_Detection.Deep_Learning.YoloV3 import Yolov3_Detection
from Tracking.Centroid import centroid_tracking
from Misc import visualise_vehicles

Yolov3_Detection.detect()
centroid_tracking.track()
#Filtering goes here
visualise_vehicles.visualise()
