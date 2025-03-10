#!/usr/bin/env python3
from enum import Enum
import argparse

from spatialmath.base import rotz
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

# trajectory parameters
CONVERGENCE_TOL = 1e-2
MIN_DURATION = 2.0  # seconds
HOME_POSE = np.array([0, 0, 0])  # TODO

# TODO basically the linear target can change depending on if hand is up, but
# angular should always keep trying to track you
# TODO how to handle multiple people?
# - need to track hand-up persistence


# TODO maybe have difference range fields for different motions


class SystemMode(Enum):
    HOME = 0
    MOVING_HOME = 1
    FOLLOWING_TARGET = 2
    UNSAFE = 3


class Person:
    def __init__(self, id, center, keypoints):
        self.id = id
        self.keypoints = keypoints
        self.center = center

    def update(self, center, keypoints):
        # only update keypoints with sufficient confidence score
        # TODO: this only works well if we have a decent confidence score at
        # some point
        mask = keypoints[:, 2] >= KPT_CONFIDENCE
        self.keypoints[mask, :] = (
            FILTER_WEIGHT * self.keypoints[mask, :]
            + (1 - FILTER_WEIGHT) * keypoints[mask, :]
        )

        self.center = FILTER_WEIGHT * self.center + (1 - FILTER_WEIGHT) * center

    def has_hand_raised(self):
        """Check if the person has their hand raised above the shoulder."""
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
        self.people = {}
        self.target_id = None

        self.safe_to_move = False

    def scan_cb(self, scan):
        """Get ranges and angles from a scan."""
        MAX_DIST = 1.0
        MIN_ANGLE = -np.pi / 4.0
        MAX_ANGLE = np.pi / 4.0

        # TODO replace with a different shape
        n = len(scan.ranges)
        ranges = np.array(scan.ranges)
        angles = np.array([scan.angle_min + i * scan.angle_increment for i in range(n)])
        valid_angles = (angles >= MIN_ANGLE) & (angles <= MAX_ANGLE)
        valid_ranges = (ranges >= scan.range_min) & (ranges <= MAX_DIST)
        valid = valid_angles & valid_ranges
        self.safe_to_move = not np.any(valid)

    # def points_callback(self, msg):
    #     if self.target is None:
    #         return
    #
    #     # scale from network image size to camera image size
    #     scale = (msg.width / MODEL_RGB_IMAGE_WIDTH, msg.height / MODEL_RGB_IMAGE_HEIGHT)
    #     uv = (self.target * scale).astype(int).tolist()
    #
    #     # get corresponding 3D points
    #     # TODO how to do this reliably?
    #     target3d = pc2.read_points_list(msg, field_names=["x", "y", "z"], uvs=[uv])[0]
    #     self.target3d = np.array([target3d.x, target3d.y, target3d.z])
    #     # print(self.target3d)

    def rgb_callback(self, msg):
        rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.rgb_image = cv2.resize(rgb_image, MODEL_RGB_IMAGE_SIZE)

        # only detect people
        detections = self.det_model.predict(
            self.rgb_image, classes=[0], max_det=1, verbose=False
        )

        # keypoint detector is really slow if no one is in the scene, so we
        # first just run detection
        if len(detections[0].boxes.cls) == 0:
            return

        results = self.model.track(self.rgb_image, verbose=False)
        boxes = results[0].boxes
        track_ids = boxes.id.int().cpu().tolist()
        keypoints = results[0].keypoints.data.cpu().numpy()

        # remove people that are no longer detected
        for id in self.people.keys():
            if id not in track_ids:
                del self.people[id]

        # add new people or update existing people
        for i, id in enumerate(track_ids):
            center = boxes.xywh[i].cpu().numpy()[:2]

            if id not in self.people:
                self.people[id] = Person(id=id, center=center, keypoints=keypoints[i])
            else:
                self.people[id].update(center=center, keypoints=keypoints[i])

        # check for raised hand
        # current target still valid: it exists and still has a raised hand
        if self.target_id is not None and self.target_id in self.people:
            if self.people[self.target_id].has_hand_raised():
                self.target = self.people[self.target_id].center
                return

        # target is invalid: check if a new target exists
        for id, person in self.people.items():
            if person.has_hand_raised():
                self.target_id = id
                self.target = person.center
                return

        # otherwise we have no target
        self.target = None

    def compute_angular_error(self):
        if self.target is None:
            error = 0
        else:
            error = MODEL_RGB_IMAGE_WIDTH / 2 - self.target[0]

            # normalize to [-1, 1]
            error /= MODEL_RGB_IMAGE_WIDTH / 2
        return error

    def annotated_image(self):
        annotator = Annotator(self.rgb_image)
        if self.keypoints is not None:
            annotator.kpts(self.keypoints)
        return annotator.result()


def change_velocity(v, vd, max_a, dt):
    """Accelerate to a desired velocity, with limits."""
    error = vd - v
    new_v = v + dt * np.sign(error) * max_a
    new_error = vd - new_v

    crossed_vd = np.sign(error) != np.sign(new_error)

    v = new_v
    v[crossed_vd] = vd[crossed_vd]
    return v


def decelerate(v, max_a, dt):
    return change_velocity(v=v, vd=np.zeros_like(v), max_a=max_a, dt=dt)


def at_home(q):
    return np.linalg.norm(home - robot.q) <= CONVERGENCE_TOL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run", action="store_true", help="Don't send commands to the robot."
    )
    parser.add_argument(
        "--display", action="store_true", help="Display the annotated image."
    )
    args = parser.parse_args()

    rospy.init_node("serving_node", disable_signals=True)
    node = ServingNode()

    rate = rospy.Rate(RATE)
    dt = 1.0 / RATE
    Kω = 0.5  # angular gain
    Kp = 1.0  # linear gain

    robot = mm.RidgebackROSInterface()
    signal_handler = mm.RobotSignalHandler(robot, args.dry_run)

    # wait until robot feedback has been received
    print("Waiting for robot...")
    while not rospy.is_shutdown() and not robot.ready():
        rate.sleep()
    print("...robot ready.")

    mode = SystemMode.HOME
    trajectory = None
    lin_vel = np.zeros(2)
    ang_vel = 0

    t0 = rospy.Time.now().to_sec()
    while not rospy.is_shutdown():
        t = rospy.Time.now().to_sec() - t0
        q = robot.q

        if args.display and node.rgb_image is not None:
            image = node.annotated_image()
            cv2.imshow("image", image)
            if cv2.waitKey(1) & 0xFF == ord(" "):
                break

        # select current mode
        if not node.safe_to_move:
            mode = SystemMode.UNSAFE
        elif node.target is not None:
            mode = SystemMode.FOLLOWING_TARGET
        elif at_home(q):
            mode = SystemMode.HOME
        else:
            mode = SystemMode.MOVING_HOME

        # if not already going home, reset trajectory
        if mode != SystemMode.MOVING_HOME:
            trajectory = None

        # move based on mode
        if mode == SystemMode.UNSAFE:
            # decelerate to zero velocity as fast as possible
            lin_vel = decelerate(lin_vel, LIN_ACC, dt)
            ang_vel = 0
        elif mode == SystemMode.HOME:
            lin_vel = np.zeros(2)
            ang_vel = 0
        elif mode == SystemMode.MOVING_HOME:
            # generate a trajectory to home pose
            if trajectory is None:
                trajectory = mm.PointToPointTrajectory.quintic(
                    q, HOME_POSE, LIN_VEL_MAX, LIN_ACC, min_duration=MIN_DURATION
                )

            # follow it (until another state is triggered)
            qd, vd, _ = trajectory.sample(t)
            error = qd - q
            cmd_vel = Kp * error + vd

            # rotate into the body frame
            C_bw = rotz(-q[2])
            cmd_vel = C_bw @ cmd_vel

        elif mode == SystemMode.FOLLOWING_TARGET:
            # move forward
            # TODO could adjust this based on angular error
            lin_vel = change_velocity(lin_vel, [LIN_VEL_MAX, 0], LIN_ACC, dt)

            # TODO apply acceleration as well
            ang_err = node.compute_angular_error()
            ang_vel = Kω * ang_err
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # enforce velocity limits
        lin_vel = np.clip(lin_vel, -LIN_VEL_MAX, LIN_VEL_MAX)
        ang_vel = np.clip(ang_vel, -ANG_VEL_MAX, ANG_VEL_MAX)

        # send command to the robot
        cmd_vel = lin_vel.append(ang_vel)
        if args.dry_run:
            print(f"cmd_vel = {cmd_vel}")
        else:
            robot.publish_cmd_vel(cmd_vel, bodyframe=True)

        rate.sleep()

    if not args.dry_run:
        robot.brake()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
