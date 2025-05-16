#!/usr/bin/env python3
from enum import Enum
import argparse

from spatialmath.base import rotz
from ultralytics import YOLO
import cv2
import rospy
import rospkg
from sensor_msgs.msg import Image, PointCloud2, LaserScan
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import ros_numpy
import serving_demo as sd
from qpsolvers import solve_qp

import mobile_manipulation_central as mm

import IPython

# TODO maybe have difference range fields for different motions

MODEL_RGB_IMAGE_WIDTH = 640
MODEL_RGB_IMAGE_HEIGHT = 480
MODEL_RGB_IMAGE_SIZE = (MODEL_RGB_IMAGE_WIDTH, MODEL_RGB_IMAGE_HEIGHT)

# control rate (Hz)
RATE = 100

# only display a new image after this much time has elapsed
DISPLAY_TIME_INTERVAL = 0.1

# base motion limits
ANG_VEL_MAX = 0.2
ANG_ACC = 0.25
LIN_VEL_MAX = 0.3
LIN_ACC = 0.15

# maximum number of detected people
MAX_PEOPLE = 5

# detection confidence
DET_CONFIDENCE = 0.5

# weight for exponential filtering of the image detections
# FILTER_WEIGHT = 0.25

MINIMUM_DEPTH = 0.25

# for home pose
CONVERGENCE_TOL = 1e-2

# time to wait when serving someone
# TODO this is way too much time
STABILIZE_TIME = 10
WAIT_TIME = 4
SERVING_TIME = STABILIZE_TIME + WAIT_TIME

# arm joint limits
Q_ELBOW_MIN = np.deg2rad(45)
Q_ELBOW_MAX = np.deg2rad(135)
Q_WRIST_MIN = -np.deg2rad(45)
Q_WRIST_MAX = np.deg2rad(45)
V_UP_MAX = 0.1  # rad/s  # TODO: tune


class SystemMode(Enum):
    HOME = 0
    MOVING_HOME = 1
    FOLLOWING_TARGET = 2
    SERVING = 3


class Person:
    def __init__(self, cls, contours):
        self.hand_up = cls == 0

        # flip because width = columns and height = rows
        self.mask = np.zeros(np.flip(MODEL_RGB_IMAGE_SIZE), dtype=np.uint8)
        cv2.drawContours(self.mask, [contours], -1, 1, cv2.FILLED)
        self.mask = self.mask.astype(bool)

        xs, ys = np.where(self.mask.T)
        x = np.median(xs)

        # we choose a lower quantile for y because we want to aim closer to the
        # head (for servoing in the z-direction)
        y = np.quantile(ys, 0.2)
        self.center = np.array([x, y], dtype=np.int32)


def compute_range_limits(angles, angle_limit=np.pi / 2, front_limit=1, side_limit=0.5):
    a = (side_limit - front_limit) / (np.cos(angle_limit) - 1)
    b = front_limit - a
    return a * np.cos(angles) + b


class ServingNode:
    def __init__(self):
        self.model = YOLO("../models/custom/weights/last.pt")
        self.bridge = CvBridge()

        self.scan_sub = rospy.Subscriber(
            "/front/scan", LaserScan, self._scan_cb, queue_size=1
        )
        self.rgb_sub = rospy.Subscriber(
            "/camera/color/image_raw", Image, self._rgb_cb, queue_size=1
        )
        self.points_sub = rospy.Subscriber(
            "/camera/depth_registered/points",
            PointCloud2,
            self._points_cb,
            queue_size=1,
        )

        self.rgb_image = None
        self.target = None
        self.target_depth = None
        self.people = []
        self.points = []

    def _scan_cb(self, scan):
        """Get ranges and angles from a scan."""

        # TODO tune this
        lidar_position = np.array([0.25, 0])

        # construct the raw points
        n = len(scan.ranges)
        ranges = np.array(scan.ranges)
        angles = np.array([scan.angle_min + i * scan.angle_increment for i in range(n)])
        points = (np.vstack((np.cos(angles), np.sin(angles))) * ranges).T

        # remove points at invalid angles
        valid = (angles >= MIN_ANGLE) & (angles <= MAX_ANGLE)
        points = points[valid, :]

        # relative to the base reference frame
        self.points = points + lidar_position
        return

        # TODO now I need to compute the normal associated with each point

        ####
        MAX_DIST = 1.25
        MIN_ANGLE = -np.pi / 4.0
        MAX_ANGLE = np.pi / 4.0

        n = len(scan.ranges)
        ranges = np.array(scan.ranges)
        angles = np.array([scan.angle_min + i * scan.angle_increment for i in range(n)])
        points = (np.vstack((np.cos(angles), np.sin(angles))) * ranges).T

        # get the returns we care about
        range_limits = compute_range_limits(
            angles,
            front_limit=MAX_DIST,
            side_limit=0.6 * MAX_DIST,
            angle_limit=MAX_ANGLE,
        )
        valid_angles = (angles >= MIN_ANGLE) & (angles <= MAX_ANGLE)
        valid_ranges = (ranges >= scan.range_min) & (ranges <= range_limits)
        valid = valid_angles & valid_ranges

        # only care about the valid points
        self.points = points[valid, :]

    def _points_cb(self, msg):
        if not self.has_target():
            self.target_depth = None
            return

        data = ros_numpy.numpify(msg)
        depth = data["z"]
        depth = cv2.resize(depth, MODEL_RGB_IMAGE_SIZE)
        target_depth = np.median(depth[self.target.mask])
        if target_depth >= MINIMUM_DEPTH:
            self.target_depth = target_depth

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
            person = Person(cls, contours)
            if person.hand_up:
                hand_up_ids.append(i)
            people.append(person)

        # no one has their hand up
        n_hand_up = len(hand_up_ids)
        if n_hand_up == 0:
            self.people = people
            self.target = None
            return

        # one target, no ambiguity
        if n_hand_up == 1:
            self.people = people
            self.target = people[hand_up_ids[0]]
            return

        # multiple targets
        if self.target is not None:
            # if there is an existing target, choose that one
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
        self.people = people

    def filter_safe_velocity(self, lin_vel, ang_vel):
        # for point in self.points:
        #     # nothing to do if velocity is zero already
        #     if np.allclose(lin_vel, 0):  # and np.isclose(ang_vel, 0):
        #         break
        #
        #     # if velocity is moving toward a detected obstacle point, remove
        #     # that component
        #     n = sd.unit(point)
        #     if n @ lin_vel > 0:
        #         t = sd.orth(n)
        #         lin_vel = (lin_vel @ t) * t

        # QP formulation
        if len(self.points) == 0:
            return lin_vel, ang_vel

        # define bounding ellipsoid
        rx = 0.75
        ry = 0.5
        A = np.diag([1.0 / rx**2, 1.0 / ry**2])
        c = np.array([0.25, 0])

        # remove points outside of the collision ellipse
        points = self.points
        x = points - c
        tangents = x @ A
        valid = np.sum(x * tangents, axis=1) <= 1

        points = points[valid, :]
        tangents = tangents[valid, :]

        n = len(points)
        if n == 0:
            # none of the points are inside the ellipse
            return lin_vel, ang_vel

        P = np.eye(3)
        ξd = np.append(lin_vel, ang_vel)
        h = np.zeros(n)

        # TODO may be able to vectorize by computing all zs at once
        S = np.array([[0, -1], [1, 0]])
        zs = np.sum(tangents * (S @ points.T).T, axis=1)
        G = np.hstack((tangents, zs[:, None]))

        # G = np.zeros((n, 3))
        # for i in range(n):
        #     normal = normals[i, :]
        #     point = points[i, :]
        #     z = -normal[0] * point[1] + normal[1] * point[0]
        #     G[i, :] = np.append(normal, z)

        x = solve_qp(P=P, q=-ξd, G=G, h=h, solver="quadprog")
        if x is None:
            print("failed to solve obstacle avoidance QP")
            return np.zeros(2), 0
        return x[:2], x[2]

    def has_target(self):
        return self.target is not None

    def compute_angular_error(self):
        if self.target is None:
            return 0

        target = self.target.center
        w2 = MODEL_RGB_IMAGE_WIDTH / 2
        error = w2 - target[0]

        # normalize to [-1, 1]
        error /= w2
        return error

    def compute_height_error(self):
        if self.target is None:
            return 0

        target = self.target.center
        h2 = MODEL_RGB_IMAGE_HEIGHT / 2
        error = h2 - target[1]

        # normalize to [-1, 1]
        error /= h2
        return error

    def annotated_image(self):
        image = self.rgb_image.copy()
        for person in self.people:
            if person.hand_up:
                image[person.mask, :] = 0
            else:
                image[person.mask, :] = 255
            cv2.circle(image, person.center, 5, [0, 0, 255], -1)
        return image


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

    # load home position
    rospack = rospkg.RosPack()
    sd_path = rospack.get_path("serving_demo")
    home = mm.load_home_position(name="default", path=sd_path + "/config/home.yaml")

    rospy.init_node("serving_node", disable_signals=True)
    node = ServingNode()

    rate = rospy.Rate(RATE)
    dt = 1.0 / RATE
    Kω = 0.5  # angular gain
    Kp = 1.0  # linear gain
    Kz = 0.5  # vertical gain

    robot = mm.MobileManipulatorROSInterface()
    tray = mm.ViconObjectInterface("ThingRoundTray")
    signal_handler = mm.RobotSignalHandler(robot, args.dry_run)

    model = mm.MobileManipulatorKinematics(tool_link_name="ur10_arm_tool0")
    stabilizer = sd.PendulumStabilizer(gain=0.5, model=model)

    # wait until robot feedback has been received
    print("Waiting for robot...")
    while not rospy.is_shutdown() and not robot.ready():
        rate.sleep()
    print("...robot ready.")

    mode = SystemMode.HOME
    lin_vel = np.zeros(2)
    ang_vel = 0
    cmd_vel = np.zeros_like(robot.q)
    last_display_time = 0

    # time at which the current mode started
    mode_start_time = 0

    t0 = rospy.Time.now().to_sec()
    while not rospy.is_shutdown():
        t = rospy.Time.now().to_sec() - t0
        mode_t = t - mode_start_time
        q = robot.q

        if (
            args.display
            and node.rgb_image is not None
            and t - last_display_time >= DISPLAY_TIME_INTERVAL
        ):
            last_display_time = t
            image = node.annotated_image()
            cv2.imshow("image", image)
            if cv2.waitKey(1) & 0xFF == ord(" "):
                break

        prev_mode = mode

        # select current mode
        if mode == SystemMode.SERVING and mode_t <= SERVING_TIME:
            # stay in serving mode until time is up
            pass
        elif (
            mode == SystemMode.FOLLOWING_TARGET
            and node.target_depth is not None
            and node.target_depth <= 1
        ):
            stabilizer.reset()
            mode = SystemMode.SERVING
        elif node.has_target():
            mode = SystemMode.FOLLOWING_TARGET
        elif np.linalg.norm(home[:2] - q[:2]) <= CONVERGENCE_TOL:
            mode = SystemMode.HOME
        elif mode != SystemMode.HOME:
            # don't switch to moving home when already at home: this can be
            # triggered by small amounts of noise in the position estimate
            mode = SystemMode.MOVING_HOME

        # mode switch
        if mode != prev_mode:
            mode_start_time = t
            print(f"mode = {mode}")
        prev_mode = mode

        # move based on mode
        lin_vel_des = np.zeros(2)
        ang_vel_des = 0
        arm_cmd_vel = np.zeros(6)

        if mode == SystemMode.HOME:
            arm_q_err = home[3:] - q[3:]
            # if mode_t <= STABILIZE_TIME:
            #     x = stabilizer.update(q, tray.position, dt)
            #     if x is None:
            #         print("failed to solve QP")
            #         break
            #     arm_cmd_vel = x[3:]
            # elif np.linalg.norm(arm_q_err) > CONVERGENCE_TOL:
            #     # move arm back to home after stabilizing
            #     arm_cmd_vel = Kp * arm_q_err
        elif mode == SystemMode.SERVING:
            # TODO this now includes the deceleration time
            if not (np.allclose(lin_vel, 0) and np.isclose(ang_vel, 0)):
                pass
                # keep pushing back the start time until base has stopped
                # serving_start = t
            # elif mode_t <= STABILIZE_TIME:
            #     x = stabilizer.update(q, tray.position, dt)
            #     if x is None:
            #         print("failed to solve QP")
            #         break
            #     arm_cmd_vel = x[3:]
        elif mode == SystemMode.MOVING_HOME:
            error = home[:3] - q[:3]
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
            arm_cmd_vel = servo_arm_up(q[3:], vz_des, dt)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # if not node.safe_to_move:
        #     # stop linear motion if it is in forward direction
        #     if lin_vel_des[0] >= 0:
        #         lin_vel_des = np.zeros(2)
        #     ang_vel_des = 0
        lin_vel_des, ang_vel_des = node.filter_safe_velocity(lin_vel_des, ang_vel_des)

        # accelerate toward desired velocity
        lin_vel = change_velocity(lin_vel, lin_vel_des, LIN_ACC, dt)
        ang_vel = change_velocity(ang_vel, ang_vel_des, ANG_ACC, dt)

        # enforce velocity limits
        lin_vel_norm = np.linalg.norm(lin_vel)
        if lin_vel_norm > LIN_VEL_MAX:
            lin_vel = LIN_VEL_MAX * lin_vel / lin_vel_norm
        ang_vel = np.clip(ang_vel, -ANG_VEL_MAX, ANG_VEL_MAX)

        # TODO should I enforce arm velocity limits as well?

        # build the full command
        base_cmd_vel = np.append(lin_vel, ang_vel)
        if args.arm_only:
            base_cmd_vel = np.zeros(3)
        cmd_vel = np.concatenate((base_cmd_vel, arm_cmd_vel))

        # send command to the robot
        if args.dry_run:
            # print(f"q = {q}")
            # print(f"cmd_vel = {cmd_vel}")
            pass
        else:
            robot.publish_cmd_vel(cmd_vel, bodyframe=True)

        rate.sleep()

    if not args.dry_run:
        robot.brake()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
