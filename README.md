# Drone Data Processing

This is a collection of python scripts used to extract trajectories from aerial drone videos of vehicles. Developed for vehicles driving on the M40 at Junction 15 however, it should be general enough to be used on all vehicles that are orientated horizontally in frame. It could also be adapted for any aerial videos of vehicles

![IMG](example.jpg)

## Requirements

The conda env file is included in the repo.

Object detection is done through YoloV3. To install the darknet framework for Yolo follow this link: [darknet](https://github.com/AlexeyAB/darknet)

## Usage
In order to process a video file, follow this procedure:

Step 1: Run main.py

Step 2: Select darknet root folder, video and detections folder. Wait for detections to finish.

Step 3: Select video, detections csv and tracked folder. Wait for tracking to finish.

Step 4: Select tracked csv and filtered folder. Wait for filtering to finish. (This doesn't work properly yet)

Step 5: Select video and file to visualise(detection, tracked or filtered).

Step 6: Use manual_correction_tool.py to correct false and missed trajectories. (doesn't work perfectly yet, new trajectory tracking has an issue)

