import cv2
import time

cap = cv2.VideoCapture("rtsp://admin:admin@192.168.100.99:554/av0_0")
while True:
    ret, image_np = cap.read()
    time.sleep(0.01)
    cv2.imshow('object detection', image_np)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
