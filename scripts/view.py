
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2
import numpy as np

import IPython

model = YOLO("yolo11s-pose.pt")

image = cv2.imread("twopeople.jpg")
image = cv2.resize(image, (640, 480))

result = model.track(image, verbose=True)

boxes = result[0].boxes
track_ids = boxes.id.int().cpu().tolist()
keypoints = result[0].keypoints.data

annotator = Annotator(image)

for box, track_id in zip(boxes, track_ids):
    b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
    c = box.cls
    annotator.box_label(b, str(track_id))

IPython.embed()

for kpts in keypoints:
    annotator.kpts(kpts)

cv2.imshow("image", annotator.result())

while cv2.waitKey(1) != ord("q"):
    pass
cv2.destroyAllWindows()
