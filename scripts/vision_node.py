#!/usr/bin/env python3
from enum import Enum
import argparse

from ultralytics import YOLO
import cv2
import rospy
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import ros_numpy

import serving_demo as sd
from serving_demo.msg import Target

import IPython


MODEL_RGB_IMAGE_WIDTH = 640
MODEL_RGB_IMAGE_HEIGHT = 480
MODEL_RGB_IMAGE_SIZE = (MODEL_RGB_IMAGE_WIDTH, MODEL_RGB_IMAGE_HEIGHT)

# control rate (Hz)
RATE = 20

# only display a new image after this much time has elapsed
DISPLAY_TIME_INTERVAL = 0.1

# maximum number of detected people
MAX_PEOPLE = 5

# detection confidence
DET_CONFIDENCE = 0.5

# weight for exponential filtering of the image detections
# FILTER_WEIGHT = 0.25

# minimum valid depth reading of the camera
MINIMUM_DEPTH = 0.25


class Person:
    def __init__(self, hand_up=False, center=None):
        self.hand_up = hand_up

        if center is None:
            center = np.zeros(2, dtype=np.int32)
        self.center = center

        self.depth = 0
        self._depth_computed = False

    @classmethod
    def from_contours(cls, class_label, contours):
        hand_up = class_label == 0

        # flip because width = columns and height = rows
        self.mask = np.zeros(np.flip(MODEL_RGB_IMAGE_SIZE), dtype=np.uint8)
        cv2.drawContours(self.mask, [contours], -1, 1, cv2.FILLED)
        self.mask = self.mask.astype(bool)

        xs, ys = np.where(self.mask.T)
        x = np.median(xs)

        # we choose a lower quantile for y because we want to aim closer to the
        # head (for servoing in the z-direction)
        y = np.quantile(ys, 0.25)
        center = np.array([x, y], dtype=np.int32)

        return cls(hand_up=hand_up, center=center)

    def compute_depth(self, pc_depth):
        depth = cv2.resize(pc_depth, MODEL_RGB_IMAGE_SIZE)
        depth = depth[self.target.mask]
        depth = depth[depth >= MINIMUM_DEPTH]
        if depth.size > 0:
            self._depth_computed = True
            self.depth = np.median(depth)
            print(f"depth = {self.depth}")

    def active(self):
        return self.hand_up and self._depth_computed


class VisionNode:
    def __init__(self):
        self.model = YOLO("../models/custom/weights/last.pt")
        self.bridge = CvBridge()

        self.rgb_image = np.zeros((MODEL_RGB_IMAGE_HEIGHT, MODEL_RGB_IMAGE_WIDTH, 3))
        self.target = Person()
        self.people = []

        self.target_pub = rospy.Publisher("/serving/target", Target, queue_size=1)

        self.rgb_sub = rospy.Subscriber(
            "/camera/color/image_raw", Image, self._rgb_cb, queue_size=1
        )
        self.pointcloud_sub = rospy.Subscriber(
            "/camera/depth_registered/points",
            PointCloud2,
            self._pointcloud_cb,
            queue_size=1,
        )

    def _pointcloud_cb(self, msg):
        data = ros_numpy.numpify(msg)
        self.target.compute_depth(data["z"])

    def _rgb_cb(self, msg):
        rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.rgb_image = cv2.resize(rgb_image, MODEL_RGB_IMAGE_SIZE)

        results = self.model.predict(
            self.rgb_image, max_det=MAX_PEOPLE, conf=DET_CONFIDENCE, verbose=False
        )

        # TODO we may want to do some filtering here
        n_det = len(results[0].boxes.cls)
        hand_up_ids = []
        people = []

        for i in range(n_det):
            cls = results[0].boxes.cls[i].cpu().numpy()
            contours = np.int32([results[0].masks.xy[i]])
            person = Person.from_contours(cls, contours)
            if person.hand_up:
                hand_up_ids.append(i)
            people.append(person)
        self.people = people

        # no one has their hand up
        n_hand_up = len(hand_up_ids)
        if n_hand_up == 0:
            # deactivate the current target
            self.target.hand_up = False
            return

        if n_hand_up == 1:
            # one target, no ambiguity
            self.target = people[hand_up_ids[0]]
            return

        # multiple targets
        if self.target.hand_up:
            # if there is an existing target, choose the one closest to the
            # existing one
            dists = np.array(
                [
                    np.linalg.norm(self.target.center - people[i].center)
                    for i in hand_up_ids
                ]
            )
            target_id = hand_up_ids[np.argmin(dists)]
        else:
            # otherwise, just pick the first one
            target_id = hand_up_ids[0]
        self.target = people[target_id]

    def annotated_image(self):
        """Produce an RGB image annotated with segmentation masks and centroids."""
        # copy to avoid corrupting the underlying image
        image = self.rgb_image.copy()
        for person in self.people:
            if person.hand_up:
                image[person.mask, :] = 0
            else:
                image[person.mask, :] = 255
            cv2.circle(image, person.center, 5, [0, 0, 255], -1)
        return image

    def publish_target(self):
        """Publish info about the target observed with the RGB-D camera."""
        msg = Target()
        msg.active = self.target.active()
        msg.x = self.target.center[0]
        msg.y = self.target.center[1]
        msg.depth = self.target.depth
        self.target_pub.publish(msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--display", action="store_true", help="Display the annotated image."
    )
    args = parser.parse_args()

    rospy.init_node("serving_vision_node", disable_signals=True)
    node = VisionNode()

    rate = rospy.Rate(RATE)
    dt = 1.0 / RATE

    last_display_time = 0

    t0 = rospy.Time.now().to_sec()
    t = t0
    while not rospy.is_shutdown():
        t_prev = t
        t = rospy.Time.now().to_sec() - t0
        print(f"dt = {t - t_prev}")

        if args.display and t - last_display_time >= DISPLAY_TIME_INTERVAL:
            last_display_time = t
            image = node.annotated_image()
            cv2.imshow("image", image)
            if cv2.waitKey(1) & 0xFF == ord(" "):
                break

        node.publish_target()
        rate.sleep()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
