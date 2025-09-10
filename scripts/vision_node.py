#!/usr/bin/env python3
import argparse
import datetime
from enum import Enum
from threading import Lock
import os

from ultralytics import YOLO
import cv2
import rospy
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import ros_numpy

import visual_servor as vs
from visual_servor.msg import Target

# control rate (Hz)
RATE = 25

# only display a new image after this much time has elapsed
DISPLAY_TIME_INTERVAL = 0.1

# maximum number of detected people
MAX_PEOPLE = 5

# detection confidence
DET_CONFIDENCE = 0.5

# weight for exponential filtering of the image detections
# FILTER_WEIGHT = 0.25


class VisionNode:
    def __init__(self):
        self.model = YOLO("../models/custom/weights/last.pt")
        self.bridge = CvBridge()

        self.rgb_image = np.zeros(
            (vs.MODEL_RGB_IMAGE_HEIGHT, vs.MODEL_RGB_IMAGE_WIDTH, 3)
        )
        self.target = vs.Person()
        self.target_lock = Lock()
        self.people = []

        # self.video_writer = cv2.VideoWriter("test.avi", -1, 

        self.target_pub = rospy.Publisher("/serving/target", Target, queue_size=1)
        # self.annotated_img_pub = rospy.Publisher(
        #     "/serving/annotated_image", Image, queue_size=1
        # )

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
        if self.target.hand_up:
            data = ros_numpy.numpify(msg)
            self.target.update_depth(data["z"])

    def _rgb_cb(self, msg):
        rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.rgb_image = cv2.resize(rgb_image, vs.MODEL_RGB_IMAGE_SIZE)

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
            person = vs.Person.from_contours(cls, contours)
            if person.hand_up:
                hand_up_ids.append(i)
            people.append(person)
        self.people = people

        # no one has their hand up
        n_hand_up = len(hand_up_ids)
        if n_hand_up == 0:
            # deactivate the current target
            self.target.hand_up = False
            self.target.depth_valid = False
            return

        if n_hand_up == 1:
            # one target, no ambiguity
            self.target = people[hand_up_ids[0]]
            return

        print("multiple hands up")

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

        # add center of target as well
        if self.target.hand_up:
            cv2.circle(image, self.target.center, 10, [0, 255, 0], -1)

        # publish for logging purposes
        # img_msg = self.bridge.cv2_to_imgmsg(image, encoding="passthrough")
        # self.annotated_img_pub.publish(img_msg)

        return image

    def publish_target(self):
        """Publish info about the target observed with the RGB-D camera."""
        # TODO use a lock to ensure consistent state?
        msg = Target()
        msg.hand_up = self.target.hand_up
        msg.x = self.target.center[0]
        msg.y = self.target.center[1]
        msg.depth_valid = self.target.depth_valid
        msg.depth = self.target.depth
        if self.target.depth_valid and self.target.depth < 1:
            print(
                f"depth valid but depth = {self.target.depth}, hand_up = {self.target.hand_up}"
            )
        self.target_pub.publish(msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--display", action="store_true", help="Display the annotated image."
    )
    parser.add_argument(
        "--save", action="store_true", help="Save the annotated images."
    )
    args = parser.parse_args()

    rospy.init_node("serving_vision_node", disable_signals=True)
    node = VisionNode()

    rate = rospy.Rate(RATE)
    dt = 1.0 / RATE

    if args.save:
        stamp = datetime.datetime.now()
        ymd = stamp.strftime("%Y-%m-%d")
        hms = stamp.strftime("%H-%M-%S")
        img_count = 1
        img_path = f"images/{ymd}_{hms}"
        os.mkdir(img_path)
        print(f"saving images to {img_path}")

    last_display_time = 0

    t0 = rospy.Time.now().to_sec()
    t_prev = 0
    t = 0
    while not rospy.is_shutdown():
        now = rospy.Time.now()
        t_prev = t
        t = now.to_sec() - t0
        # print(f"dt = {t - t_prev}")

        if args.display:  # and t - last_display_time >= DISPLAY_TIME_INTERVAL:
            last_display_time = t
            image = node.annotated_image()

            # save the image
            if args.save:
                cv2.imwrite(f"{img_path}/annotated_{img_count}_{now.to_nsec()}.png", image)
                img_count += 1

            cv2.imshow("image", image)
            if cv2.waitKey(1) & 0xFF == ord(" "):
                break

        node.publish_target()
        rate.sleep()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
