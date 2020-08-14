import cv2
from darknet_video import YOLO

YOLO(video_path = 'town.mp4', configPath = "cfg/custom-yolov4-detector.cfg", weightPath = "weights/yolo_3000.weights", metaPath = "data/obj.data")
