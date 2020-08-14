#!/bin/sh
echo 'Welcome to CoVidSpy'
sleep 2
echo 'Enter the path of the video to test on'
read video_dir

./darknet detector demo data/obj.data cfg/custom-yolov4-detector.cfg weights/yolo_3000.weights $video_dir -dont_show -out_filename res.avi

python3 play_video.py

