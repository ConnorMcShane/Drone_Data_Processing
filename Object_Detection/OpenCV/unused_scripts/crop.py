import cv2
img = cv2.imread(r"C:\Users\cmcshan1\Documents\DroneFootage\Drone_Videos\stabilized\frames\DJI_0059_backup\uncropped\stabilized_image_06900.jpg")
h, w, c = img.shape
crop_img = img[int(h*0.05):int(h*0.95), int(w*0.05):int(w*0.95)]
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)