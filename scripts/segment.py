import time
from ultralytics import YOLO
from ultralytics.models.fastsam import FastSAMPredictor
from ultralytics.utils.plotting import Annotator
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image, PointCloud2, LaserScan
from cv_bridge import CvBridge
import ros_numpy

import IPython

MODEL_RGB_IMAGE_WIDTH = 640
MODEL_RGB_IMAGE_HEIGHT = 480
MODEL_RGB_IMAGE_SIZE = (MODEL_RGB_IMAGE_WIDTH, MODEL_RGB_IMAGE_HEIGHT)


class Person:
    def __init__(self):
        self.mask = np.zeros(MODEL_RGB_IMAGE_SIZE, dtype=np.uint8)
        self.center = None
        self.hand_up = False

    @property
    def visible(self):
        return self.center is not None

    def update(self, contours, class):
        self.hand_up = class == 0
        cv2.drawContours(self.mask, [contours], -1, 1, cv2.FILLED)
        self.center = np.mean(np.where(self.mask.T == 1), axis=1).astype(np.int32)


class CameraNode:
    def __init__(self):
        # self.model = YOLO("../models/yolo11n-seg.pt")
        self.model = YOLO("../models/custom/weights/last.pt")
        self.bridge = CvBridge()

        self.rgb_sub = rospy.Subscriber(
            "/camera/color/image_raw", Image, self.rgb_callback, queue_size=1
        )
        self.points_sub = rospy.Subscriber(
            "/camera/depth_registered/points",
            PointCloud2,
            self.points_callback,
            queue_size=1,
        )

        self.rgb_image = None
        self.person = None

    def points_callback(self, msg):
        if self.person is None:
            return

        # TODO get the actual depth data
        data = ros_numpy.numpify(msg)
        depth = data["z"]
        depth = cv2.resize(depth, MODEL_RGB_IMAGE_SIZE)
        print(np.median(depth[self.person.mask]))

    def rgb_callback(self, msg):
        rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        rgb_image = cv2.resize(rgb_image, MODEL_RGB_IMAGE_SIZE)

        results = self.model.predict(rgb_image, max_det=1, conf=0.5, verbose=False)

        # nothing to do if there are no detections
        if len(results[0].boxes.cls) == 0:
            return
        # print(results[0].boxes.cls[0])

        contours = np.int32([results[0].masks.xy[0]])
        self.person = Person(contours)

        if results[0].boxes.cls[0] == 0:
            rgb_image[self.person.mask, :] = 0
        else:
            rgb_image[self.person.mask, :] = 255
        cv2.circle(rgb_image, self.person.center, 5, [0, 0, 255], -1)
        self.rgb_image = rgb_image

    def annotated_image(self):
        annotator = Annotator(self.rgb_image)

        # (shallow) copy to avoid changing dict size during iteration
        people = self.people.copy()
        for track_id, person in people.items():
            annotator.kpts(person.keypoints)
            annotator.box_label(person.box_xyxy, str(track_id))
        return annotator.result()


def main():
    rospy.init_node("camera_node")
    node = CameraNode()
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        if node.rgb_image is not None:
            cv2.imshow("image", node.rgb_image)
            if cv2.waitKey(1) & 0xFF == ord(" "):
                break
        rate.sleep()


if __name__ == "__main__":
    main()

# # Create FastSAMPredictor
# overrides = dict(
#     conf=0.25,
#     task="segment",
#     mode="predict",
#     model="FastSAM-s.pt",
#     save=False,
#     imgsz=640,
# )
# predictor = FastSAMPredictor(overrides=overrides)
#
# # Segment everything
# everything_results = predictor("twopeople.jpg")
#
# # Prompt inference
# text_results = predictor.prompt(everything_results, texts="a person")
