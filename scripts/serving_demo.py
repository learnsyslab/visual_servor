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

# TODO maybe have difference range fields for different motions

MODEL_RGB_IMAGE_WIDTH = 640
MODEL_RGB_IMAGE_HEIGHT = 480
MODEL_RGB_IMAGE_SIZE = (MODEL_RGB_IMAGE_WIDTH, MODEL_RGB_IMAGE_HEIGHT)

# control rate (Hz)
RATE = 10

# base motion limits
ANG_VEL_MAX = 0.2
ANG_ACC = 0.15
LIN_VEL_MAX = 0.3
LIN_ACC = 0.15

# threshold for detected keypoint confidence
KPT_CONFIDENCE = 0.75

# detection confidence
DET_CONFIDENCE = 0.5

# weight for exponential filtering of the image detections
FILTER_WEIGHT = 0.25

# home pose
CONVERGENCE_TOL = 1e-2
HOME_POSE = np.array([-2, -1, -np.pi / 4])

# time to wait when serving someone
SERVING_TIME = 6

# arm joint limits
Q_ELBOW_MIN = np.deg2rad(75)
Q_ELBOW_MAX = np.deg2rad(120)
Q_WRIST_MIN = 0
Q_WRIST_MAX = np.deg2rad(45)
V_UP_MAX = 0.1  # rad/s  # TODO: tune


class SystemMode(Enum):
    HOME = 0
    MOVING_HOME = 1
    FOLLOWING_TARGET = 2
    SERVING = 3


class Person:
    def __init__(self, id, center, keypoints):
        self.id = id
        self.keypoints = keypoints

        # a default center value must be provided, but we will track the head
        # if possible
        self.center = center
        head_pos = self._compute_head_position()
        if head_pos is not None:
            self.center = head_pos

        self.box_xyxy = np.zeros(4)

    def _compute_head_position(self):
        head_kpts = self.keypoints[:5, :]
        head_mask = head_kpts[:, 2] > KPT_CONFIDENCE
        if not np.any(head_mask):
            return None
        return np.mean(head_kpts[head_mask, :2], axis=0)

    def _in_box(self, keypoints):
        return (
            (keypoints[:, 0] >= self.box_xyxy[0])
            & (keypoints[:, 0] <= self.box_xyxy[2])
            & (keypoints[:, 1] >= self.box_xyxy[1])
            & (keypoints[:, 1] <= self.box_xyxy[3])
        )

    def update(self, keypoints):
        # only the confident positions are updated, but all the confidences are
        # updated
        mask = keypoints[:, 2] >= KPT_CONFIDENCE  #  & self._in_box(keypoints)
        w1 = FILTER_WEIGHT
        w2 = 1 - FILTER_WEIGHT
        self.keypoints[mask, :2] = (
            w1 * self.keypoints[mask, :2] + w2 * keypoints[mask, :2]
        )
        self.keypoints[:, 2] = w1 * self.keypoints[:, 2] + w2 * keypoints[:, 2]

        # update center (of the head)
        center = self._compute_head_position()
        if center is None:
            return
        self.center = w1 * self.center + w2 * center

    def has_hand_raised(self):
        """Check if the person has their hand raised above the shoulder."""
        # left_shoulder = self.keypoints[5, :]
        # right_shoulder = self.keypoints[6, :]
        left_wrist = self.keypoints[9, :]
        right_wrist = self.keypoints[10, :]

        head_kpts = self.keypoints[:5, :]
        head_mask = head_kpts[:, 2] > KPT_CONFIDENCE
        if not np.any(head_mask):
            return False
        head_height = np.min(head_kpts[head_mask, 1])

        # recall that this is in image coordinates, so y is flipped
        if left_wrist[2] > KPT_CONFIDENCE and left_wrist[1] < head_height:
            return True

        if right_wrist[2] > KPT_CONFIDENCE and right_wrist[1] < head_height:
            return True

        return False


def compute_range_limits(angles, angle_limit=np.pi / 2, front_limit=1, side_limit=0.5):
    a = (side_limit - front_limit) / (np.cos(angle_limit) - 1)
    b = front_limit - a
    return a * np.cos(angles) + b


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
        MAX_DIST = 1.25
        MIN_ANGLE = -np.pi / 4.0
        MAX_ANGLE = np.pi / 4.0

        n = len(scan.ranges)
        ranges = np.array(scan.ranges)
        angles = np.array([scan.angle_min + i * scan.angle_increment for i in range(n)])
        range_limits = compute_range_limits(
            angles,
            front_limit=MAX_DIST,
            side_limit=0.6*MAX_DIST,
            angle_limit=MAX_ANGLE,
        )
        valid_angles = (angles >= MIN_ANGLE) & (angles <= MAX_ANGLE)
        valid_ranges = (ranges >= scan.range_min) & (ranges <= range_limits)
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

        results = self.model.track(self.rgb_image, conf=DET_CONFIDENCE, verbose=False)
        boxes = results[0].boxes

        # if the detector is uncertain the person won't be tracked right away,
        # so we ignore for now
        if not boxes.is_track:
            return

        track_ids = boxes.id.int().cpu().tolist()
        keypoints = results[0].keypoints.data.cpu().numpy()

        # remove people that are no longer detected
        # note conversion to list because we cannot delete keys while iterating
        for id in list(self.people.keys()):
            if id not in track_ids:
                del self.people[id]

        # add new people or update existing people
        for i, id in enumerate(track_ids):
            if id not in self.people:
                # default center is centered in x and 3/4 up in y
                x, y, w, h = boxes.xywh[i].cpu().numpy()
                center = np.array([x, y - 0.25 * h])
                self.people[id] = Person(id=id, center=center, keypoints=keypoints[i])
            else:
                self.people[id].update(keypoints=keypoints[i])

            self.people[id].box_xyxy = boxes.xyxy[i].cpu().numpy()
            # print(boxes.conf[i].cpu().numpy())

        # check for raised hand
        # current target still valid: it exists and still has a raised hand
        if self.target_id is not None and self.target_id in self.people:
            if self.people[self.target_id].has_hand_raised():
                return

        # target is invalid: check if a new target exists
        for id, person in self.people.items():
            if person.has_hand_raised():
                self.target_id = id
                return

        # otherwise we have no target
        self.target_id = None

    def has_target(self):
        return self.target_id is not None

    def compute_angular_error(self):
        if self.target_id is None:
            return 0

        target = self.people[self.target_id].center
        w2 = MODEL_RGB_IMAGE_WIDTH / 2
        error = w2 - target[0]

        # normalize to [-1, 1]
        error /= w2
        return error

    def compute_height_error(self):
        if self.target_id is None:
            return 0

        target = self.people[self.target_id].center
        h2 = MODEL_RGB_IMAGE_HEIGHT / 2
        error = h2 - target[1]

        # normalize to [-1, 1]
        error /= h2
        return error

    def annotated_image(self):
        annotator = Annotator(self.rgb_image)

        # (shallow) copy to avoid changing dict size during iteration
        people = self.people.copy()
        for track_id, person in people.items():
            annotator.kpts(person.keypoints)
            annotator.box_label(person.box_xyxy, str(track_id))
        return annotator.result()


def change_velocity(v, vd, max_a, dt):
    """Accelerate to a desired velocity, with limits."""
    scalar = np.isscalar(v)
    v = np.atleast_1d(v)
    vd = np.atleast_1d(vd)

    error = vd - v
    new_v = v + dt * np.sign(error) * max_a
    new_error = vd - new_v

    crossed_vd = np.sign(error) != np.sign(new_error)

    v = new_v
    v[crossed_vd] = vd[crossed_vd]
    if scalar:
        return v[0]
    return v


def decelerate(v, max_a, dt):
    """Decelerate to zero velocity subject to maximum acceleration."""
    return change_velocity(v=v, vd=np.zeros_like(v), max_a=max_a, dt=dt)


def at_home(q):
    """Returns ``True`` if the robot is at the home position, ``False`` otherwise."""
    return np.linalg.norm(HOME_POSE[:2] - q[:2]) <= CONVERGENCE_TOL


def servo_arm_up(q, vz, dt):
    """Servo the EE up using the arm.

    Parameters
    ----------
    q : np.array, shape (6,)
        The arm joint angles.
    vz : float
        The vertical velocity.
    dt : float
        The control timestep.

    Returns
    -------
    : np.array, shape (6,)
        The commanded arm joint velocity.
    """
    assert q.shape == (6,)
    elbow_idx = 2
    wrist_idx = 3

    # limit the velocity
    vz = np.clip(vz, -V_UP_MAX, V_UP_MAX)

    cmd_vel = np.zeros_like(q)
    cmd_vel[elbow_idx] = -vz
    cmd_vel[wrist_idx] = vz

    # don't move if bounds would be violated
    q_next = q + dt * cmd_vel
    if (
        q_next[elbow_idx] <= Q_ELBOW_MIN
        or q_next[elbow_idx] >= Q_ELBOW_MAX
        or q_next[wrist_idx] <= Q_WRIST_MIN
        or q_next[wrist_idx] >= Q_WRIST_MAX
    ):
        return np.zeros_like(cmd_vel)

    return cmd_vel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run", action="store_true", help="Don't send commands to the robot."
    )
    parser.add_argument(
        "--display", action="store_true", help="Display the annotated image."
    )
    parser.add_argument("--arm-only", action="store_true", help="Only move the arm.")
    args = parser.parse_args()

    rospy.init_node("serving_node", disable_signals=True)
    node = ServingNode()

    rate = rospy.Rate(RATE)
    dt = 1.0 / RATE
    Kω = 0.5  # angular gain
    Kp = 1.0  # linear gain
    Kz = 0.5  # vertical gain

    robot = mm.MobileManipulatorROSInterface()
    signal_handler = mm.RobotSignalHandler(robot, args.dry_run)

    # wait until robot feedback has been received
    print("Waiting for robot...")
    while not rospy.is_shutdown() and not robot.ready():
        rate.sleep()
    print("...robot ready.")

    mode = SystemMode.HOME
    lin_vel = np.zeros(2)
    ang_vel = 0
    serving_start = 0

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
        if mode == SystemMode.SERVING and t - serving_start <= SERVING_TIME:
            # stay in serving mode until time is up
            pass
        elif node.has_target():
            mode = SystemMode.FOLLOWING_TARGET
        elif at_home(q):
            mode = SystemMode.HOME
        else:
            # don't switch to moving home when already at home: this can be
            # triggered by small amounts of noise in the position estimate
            if mode != SystemMode.HOME:
                mode = SystemMode.MOVING_HOME
        print(mode)

        # move based on mode
        vz_des = 0
        if mode == SystemMode.HOME or mode == SystemMode.SERVING:
            lin_vel_des = np.zeros(2)
            ang_vel_des = 0
        elif mode == SystemMode.MOVING_HOME:
            error = HOME_POSE[:3] - q[:3]
            error[2] = mm.wrap_to_pi(error[2])
            vd = Kp * error

            # rotate into the body frame
            C_bw = rotz(-q[2])
            vd = C_bw @ vd

            lin_vel_des = vd[:2]
            ang_vel_des = 0  # keep current angle

        elif mode == SystemMode.FOLLOWING_TARGET:
            # base motion
            ang_err = node.compute_angular_error()
            vx = LIN_VEL_MAX * (1 - np.abs(ang_err))
            lin_vel_des = np.array([vx, 0])
            ang_vel_des = Kω * ang_err

            # arm motion
            height_err = node.compute_height_error()
            vz_des = Kz * height_err
        else:
            raise ValueError(f"Invalid mode: {mode}")

        if not node.safe_to_move:
            # stop linear motion if it is in forward direction
            if lin_vel_des[0] >= 0:
                lin_vel_des = np.zeros(2)
            ang_vel_des = 0

            # stay still to serve
            if mode == SystemMode.FOLLOWING_TARGET:
                print("start serve")
                lin_vel_des = np.zeros(2)
                ang_vel_des = 0
                serving_start = t
                mode = SystemMode.SERVING

        # accelerate toward desired velocity
        lin_vel = change_velocity(lin_vel, lin_vel_des, LIN_ACC, dt)
        ang_vel = change_velocity(ang_vel, ang_vel_des, ANG_ACC, dt)

        # enforce velocity limits
        lin_vel_norm = np.linalg.norm(lin_vel)
        if lin_vel_norm > LIN_VEL_MAX:
            lin_vel = LIN_VEL_MAX * lin_vel / lin_vel_norm
        ang_vel = np.clip(ang_vel, -ANG_VEL_MAX, ANG_VEL_MAX)

        # build the full command
        base_cmd_vel = np.append(lin_vel, ang_vel)
        if args.arm_only:
            base_cmd_vel = np.zeros(3)

        arm_cmd_vel = servo_arm_up(q[3:], vz_des, dt)
        cmd_vel = np.concatenate((base_cmd_vel, arm_cmd_vel))

        print(f"vz_des = {vz_des}")
        if node.has_target():
            print(f"target = {node.people[node.target_id].center}")

        # send command to the robot
        if args.dry_run:
            print(f"q = {q}")
            print(f"cmd_vel = {cmd_vel}")
        else:
            robot.publish_cmd_vel(cmd_vel, bodyframe=True)

        rate.sleep()

    if not args.dry_run:
        robot.brake()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
