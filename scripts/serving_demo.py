#!/usr/bin/env python3
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2
import rospy
from sensor_msgs.msg import Image, PointCloud2, LaserScan
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
import numpy as np

import mobile_manipulation_central as mm

import IPython

MODEL_RGB_IMAGE_WIDTH = 640
MODEL_RGB_IMAGE_HEIGHT = 480
MODEL_RGB_IMAGE_SIZE = (MODEL_RGB_IMAGE_WIDTH, MODEL_RGB_IMAGE_HEIGHT)

RATE = 10  # Hz
ANG_VEL_MAX = 0.1
LIN_VEL_MAX = 0.1
LIN_ACC = 0.1
KPT_CONFIDENCE = 0.75
FILTER_WEIGHT = 0.25

DRY_RUN = False
DISPLAY = True

# TODO basically the linear target can change depending on if hand is up, but
# angular should always keep trying to track you
# TODO how to handle multiple people?


class ServingNode:
    def __init__(self):
        self.det_model = YOLO("yolo11s.pt")

        self.model = YOLO("yolo11s-pose.pt")
        self.bridge = CvBridge()

        self.scan_sub = rospy.Subscriber(
            "/front/scan", LaserScan, self.scan_cb, queue_size=1
        )
        self.rgb_sub = rospy.Subscriber(
            "/camera/color/image_raw", Image, self.rgb_callback, queue_size=1
        )
        # self.points_sub = rospy.Subscriber(
        #     "/camera/depth_registered/points",
        #     PointCloud2,
        #     self.points_callback,
        #     queue_size=1,
        # )

        self.rgb_image = None
        self.keypoints = None
        self.target = None
        # self.target3d = None
        self.safe_to_move = False
        self.n_detections = 0

    # def ready(self):
    #     return self.keypoints is not None

    def scan_cb(self, scan):
        """Get ranges and angles from a scan."""
        MAX_DIST = 1.0
        MIN_ANGLE = -np.pi / 4.0
        MAX_ANGLE = np.pi / 4.0

        n = len(scan.ranges)
        ranges = np.array(scan.ranges)
        angles = np.array([scan.angle_min + i * scan.angle_increment for i in range(n)])
        valid_angles = (angles >= MIN_ANGLE) & (angles <= MAX_ANGLE)
        valid_ranges = (ranges >= scan.range_min) & (ranges <= MAX_DIST)
        valid = valid_angles & valid_ranges
        self.safe_to_move = not np.any(valid)

    def points_callback(self, msg):
        if self.target is None:
            return

        # scale from network image size to camera image size
        scale = (msg.width / MODEL_RGB_IMAGE_WIDTH, msg.height / MODEL_RGB_IMAGE_HEIGHT)
        uv = (self.target * scale).astype(int).tolist()

        # get corresponding 3D points
        # TODO how to do this reliably?
        target3d = pc2.read_points_list(msg, field_names=["x", "y", "z"], uvs=[uv])[0]
        self.target3d = np.array([target3d.x, target3d.y, target3d.z])
        # print(self.target3d)

    def rgb_callback(self, msg):
        rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.rgb_image = cv2.resize(rgb_image, MODEL_RGB_IMAGE_SIZE)

        # only detect people
        detections = self.det_model.predict(
            self.rgb_image, classes=[0], max_det=2, verbose=False
        )

        # keypoint detector is really slow if no one is in the scene, so we
        # first just run detection
        self.n_detections = len(detections[0].boxes.cls)
        if self.n_detections != 1:
            return

        results = self.model.predict(self.rgb_image, max_det=1, verbose=False)
        keypoints = np.asarray(results[0].keypoints.data[0].cpu())

        # no keypoints detected
        if keypoints.size == 0:
            return

        # only update keypoints with sufficient confidence score
        # TODO this only works well if we have a decent confidence score at
        # some point
        if self.keypoints is None:
            self.keypoints = keypoints
        else:
            mask = keypoints[:, 2] >= KPT_CONFIDENCE
            self.keypoints[mask, :] = (
                FILTER_WEIGHT * self.keypoints[mask, :]
                + (1 - FILTER_WEIGHT) * keypoints[mask, :]
            )

        # target position is the center of the bounding box
        x, y, w, h = np.asarray(results[0].boxes.xywh[0].cpu())
        target = np.array([x + 0.5 * w, y + 0.5 * h])
        if self.target is None:
            self.target = target
        else:
            self.target = FILTER_WEIGHT * self.target + (1 - FILTER_WEIGHT) * target

    def compute_angular_error(self):
        if self.target is None:
            error = 0
        else:
            error = MODEL_RGB_IMAGE_WIDTH / 2 - self.target[0]

            # normalize to [-1, 1]
            error /= MODEL_RGB_IMAGE_WIDTH / 2
        return error

    def is_hand_raised(self):
        # assume no if there is no detection
        if self.n_detections != 1:
            return False

        if self.keypoints is None:
            return False

        left_shoulder = self.keypoints[5, :]
        right_shoulder = self.keypoints[6, :]
        left_wrist = self.keypoints[9, :]
        right_wrist = self.keypoints[10, :]

        # recall that this is in image coordinates, so y is flipped
        if left_wrist[2] > KPT_CONFIDENCE and left_shoulder[2] > KPT_CONFIDENCE:
            if left_wrist[1] < left_shoulder[1]:
                return True

        if right_wrist[2] > KPT_CONFIDENCE and right_shoulder[2] > KPT_CONFIDENCE:
            if right_wrist[1] < right_shoulder[1]:
                return True

        return False

    def annotated_image(self):
        annotator = Annotator(self.rgb_image)
        if self.keypoints is not None:
            annotator.kpts(self.keypoints)
        return annotator.result()


def main():
    rospy.init_node("serving_node", disable_signals=True)
    node = ServingNode()

    rate = rospy.Rate(RATE)
    dt = 1. / RATE
    Kω = 0.5  # NOTE: needs to be small

    robot = mm.RidgebackROSInterface()
    signal_handler = mm.RobotSignalHandler(robot, DRY_RUN)

    # wait until robot feedback has been received
    # print("Waiting for robot...")
    # while not rospy.is_shutdown() and not robot.ready():
    #     rate.sleep()
    # print("...robot ready.")
    lin_vel = 0
    while not rospy.is_shutdown():
        if DISPLAY and node.rgb_image is not None:
            image = node.annotated_image()
            cv2.imshow("image", image)
            if cv2.waitKey(1) & 0xFF == ord(" "):
                break

        angular_error = node.compute_angular_error()
        angular_cmd = Kω * angular_error

        # limit the angular velocity
        if np.abs(angular_cmd) > ANG_VEL_MAX:
            angular_cmd = np.sign(angular_cmd) * ANG_VEL_MAX

        if node.safe_to_move and node.is_hand_raised():
            lin_vel += dt * LIN_ACC
        else:
            lin_vel -= dt * LIN_ACC
        lin_vel = max(lin_vel, 0)
        lin_vel = min(lin_vel, LIN_VEL_MAX)

        cmd_vel = np.array([lin_vel, 0, angular_cmd])
        if DRY_RUN:
            print(f"cmd_vel = {cmd_vel}")
        else:
            robot.publish_cmd_vel(cmd_vel, bodyframe=True)

        rate.sleep()

    if not DRY_RUN:
        robot.brake()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
