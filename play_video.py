import numpy as np
import cv2 as cv
cap = cv.VideoCapture('res.avi')
while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    cv.imshow('Output Video', frame)
    if cv.waitKey(25) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
