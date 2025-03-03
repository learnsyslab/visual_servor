#!/usr/bin/env python3
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import Image

import mobile_manipulation_central as mm

import IPython

RGB_IMAGE_WIDTH = 640
RGB_IMAGE_HEIGHT = 480
RGB_IMAGE_SIZE = (RGB_IMAGE_WIDTH, RGB_IMAGE_HEIGHT)

DISPLAY = True


def hand_raised(keypoints, conf=0.75):
    left_shoulder = keypoints[5, :]
    right_shoulder = keypoints[6, :]
    left_wrist = keypoints[9, :]
    right_wrist = keypoints[10, :]

    # recall that this is in image coordinates, so y is flipped
    if left_wrist[2] > conf and left_shoulder[2] > conf:
        if left_wrist[1] < left_shoulder[1]:
            return left_wrist[:2]

    if right_wrist[2] > conf and right_shoulder[2] > conf:
        if right_wrist[1] < right_shoulder[1]:
            return right_wrist[:2]

    return None


class ServingNode:
    def __init__(self):
        self.model = YOLO("yolo11s-pose.pt")
        self.bridge = CvBridge()

        self.rgb_sub = rospy.Subscriber(
            "/camera/color/image_raw", Image, self.rgb_callback
        )

        self.rgb_image = None
        self.results = None

        # TODO: only want to update the values that are visible
        # TODO: actually I want a KF with a zero-velocity motion model, which
        # means that
        # self.keypoint_filter = mm.ExponentialFilter(0.5)

    def ready(self):
        return self.results is not None

    def rgb_callback(self, msg):
        rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.rgb_image = cv2.resize(rgb_image, RGB_IMAGE_SIZE)
        self.results = self.model.predict(self.rgb_image, verbose=False)

    def process_target(self):
        for points in self.results[0].keypoints.data:
            target = hand_raised(points)
            break
        error = RGB_IMAGE_WIDTH / 2 - target[0]
        return error

    def annotated_image(self):
        annotator = Annotator(self.rgb_image)
        for points in self.results[0].keypoints.data:
            annotator.kpts(points)
        return annotator.result()


def main():
    rospy.init_node("serving_node")
    node = ServingNode()

    rate = rospy.Rate(5)

    K = 0.001  # NOTE: needs to be small

    while not rospy.is_shutdown():
        if not node.ready():
            rate.sleep()
            continue

        if DISPLAY:
            image = node.annotated_image()
            cv2.imshow("image", image)
            if cv2.waitKey(1) & 0xFF == ord(" "):
                break

        angular_error = node.process_target()
        angular_cmd = K * angular_error

        # limit the angular velocity
        if np.abs(angular_cmd) > 0.5:
            angular_cmd = np.sign(angular_cmd) * 0.5

        # TODO want some acceleration limits built in too
        # TODO command rate?

        rate.sleep()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
