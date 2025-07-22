#!/usr/bin/env python3
from enum import Enum
import argparse

from spatialmath.base import rotz
import rospy
import rospkg
from sensor_msgs.msg import LaserScan
import numpy as np
from qpsolvers import solve_qp
import yaml

import mobile_manipulation_central as mm
import serving_demo as sd
from serving_demo.msg import Target


import IPython


USE_STABILIZER = True
USE_COLLISION_AVOIDANCE = True


# control rate (Hz)
RATE = 125

TARGET_TIME_DELTA_MAX = 3

# lidar offset from base origin
# TODO: tune this
LIDAR_OFFSET = np.array([0.25, 0])
NUM_COLLISION_POINTS = 20

# base motion limits
BASE_VEL_MAX = np.array([0.3, 0.3, 0.2])
BASE_ACC_MAX = np.array([0.15, 0.15, 0.25])

# for home pose
CONVERGENCE_TOL = 1e-2

# if tray velocity is below this, then we consider it stabilized
TRAY_VEL_TOL = 0.01

MIN_STABILIZE_TIME = 2
MAX_STABILIZE_TIME_HOME = 10
MAX_STABILIZE_TIME_SERVING = 6
WAIT_TIME = 4

# NOTE: the serving time includes time to decelerate
SERVING_TIME = MAX_STABILIZE_TIME_SERVING + WAIT_TIME

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


class ControlNode:
    def __init__(self):
        self.collision_ellipse = sd.CollisionEllipse(rx=0.8, ry=0.75, center=[0.25, 0])

        self.target = sd.Person()
        self.target_time_recv = rospy.Time.now().to_sec()
        self.points = []

        self.scan_sub = rospy.Subscriber(
            "/front/scan", LaserScan, self._scan_cb, queue_size=1
        )
        self.target_sub = rospy.Subscriber(
            "/serving/target", Target, self._target_cb, queue_size=1
        )

    def _target_cb(self, msg):
        self.target.hand_up = msg.hand_up
        self.target.center = np.array([msg.x, msg.y], dtype=np.int32)
        self.target.depth_valid = msg.depth_valid
        self.target.depth = msg.depth

        self.target_time_recv = rospy.Time.now().to_sec()

    def _scan_cb(self, scan):
        """Get ranges and angles from a scan."""
        if not USE_COLLISION_AVOIDANCE:
            return
        self.points = self.collision_ellipse.process_scan(
            scan, lidar_offset=LIDAR_OFFSET, num_buckets=NUM_COLLISION_POINTS
        )

    def filter_safe_velocity(self, base_vel_des):
        if not USE_COLLISION_AVOIDANCE:
            return base_vel_des
        return self.collision_ellipse.filter_safe_velocity(base_vel_des, self.points)

    def compute_angular_error(self):
        if not self.target.hand_up:
            return 0

        x = self.target.center[0]
        w2 = sd.MODEL_RGB_IMAGE_WIDTH / 2
        error = w2 - x

        # normalize to [-1, 1]
        error /= w2
        return error

    def compute_height_error(self):
        if not self.target.hand_up:
            return 0

        y = self.target.center[1]
        h2 = sd.MODEL_RGB_IMAGE_HEIGHT / 2
        error = h2 - y

        # normalize to [-1, 1]
        error /= h2
        return error


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


def limit_base_vel(base_cmd_vel):
    # enforce velocity limits
    linear, angular = base_cmd_vel[:2], base_cmd_vel[2]

    lin_norm = np.linalg.norm(linear)
    if lin_norm > BASE_VEL_MAX[0]:
        linear = BASE_VEL_MAX[0] * linear / lin_norm

    angular = np.clip(angular, -BASE_VEL_MAX[2], BASE_VEL_MAX[2])

    return np.append(linear, angular)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run", action="store_true", help="Don't send commands to the robot."
    )
    parser.add_argument("--arm-only", action="store_true", help="Only move the arm.")
    args = parser.parse_args()

    # load home position
    rospack = rospkg.RosPack()
    sd_path = rospack.get_path("serving_demo")
    home = mm.load_home_position(name="default", path=sd_path + "/config/home.yaml")

    # load pendulum calibration
    # TODO clean up?
    calib_path = sd_path + "/config/pendulum_calibration.yaml"
    with open(calib_path) as f:
        calib = yaml.safe_load(f)
        r_tray_ee = np.array(calib["r_tray_ee"])

    rospy.init_node("serving_node", disable_signals=True)
    node = ControlNode()

    rate = rospy.Rate(RATE)
    dt = 1.0 / RATE
    Kω = 0.5  # angular gain
    Kp = 1.0  # linear gain
    Kz = 0.5  # vertical gain

    robot = mm.MobileManipulatorROSInterface()
    tray = mm.ViconObjectInterface("ThingRoundTray")
    signal_handler = mm.RobotSignalHandler(robot, args.dry_run)

    model = mm.MobileManipulatorKinematics(tool_link_name="pendulum_pivot")
    stabilizer = sd.PendulumStabilizer(model=model)
    home_stabilizer_timer = sd.PendulumStabilizerTimer(
        stabilizer=stabilizer,
        min_time=MIN_STABILIZE_TIME,
        max_time=MAX_STABILIZE_TIME_HOME,
        tray_vel_tol=TRAY_VEL_TOL,
    )
    serving_stabilizer_timer = sd.PendulumStabilizerTimer(
        stabilizer=stabilizer,
        min_time=MIN_STABILIZE_TIME,
        max_time=MAX_STABILIZE_TIME_SERVING,
        tray_vel_tol=TRAY_VEL_TOL,
    )

    # wait until robot feedback has been received
    print("Waiting for robot...")
    while not rospy.is_shutdown() and not robot.ready():
        rate.sleep()
    print("...robot ready.")

    stabilizer.init(q0=robot.q, r_tray_ee=r_tray_ee)
    home_stabilizer_timer.activate()

    mode = SystemMode.HOME
    base_cmd_vel = np.zeros(3)

    # time at which the current mode started
    mode_start_time = 0

    # wrap the whole thing in try/finally to ensure we actually send a brake
    # command to the robot
    try:
        t0 = rospy.Time.now().to_sec()
        t = t0
        while not rospy.is_shutdown():
            t_prev = t
            now = rospy.Time.now().to_sec()
            t = now - t0
            # print(f"dt = {t - t_prev}")

            # stop if target is too delayed
            if now - node.target_time_recv > TARGET_TIME_DELTA_MAX:
                print(f"target not received for {now - node.target_time_recv} sec")
                break

            mode_t = t - mode_start_time
            q = robot.q
            prev_mode = mode

            # select current mode
            if mode == SystemMode.SERVING and mode_t <= SERVING_TIME:
                # stay in serving mode until time is up
                pass
            elif (
                mode == SystemMode.FOLLOWING_TARGET
                and node.target.depth_valid
                and node.target.depth <= 1.5
            ):
                print(f"target depth = {node.target.depth}")
                stabilizer.reset(q)
                serving_stabilizer_timer.activate()
                mode = SystemMode.SERVING
            elif node.target.hand_up:
                mode = SystemMode.FOLLOWING_TARGET
            elif (
                mode == SystemMode.MOVING_HOME
                and np.linalg.norm(home[:2] - q[:2]) <= CONVERGENCE_TOL
            ):
                stabilizer.reset(q)
                home_stabilizer_timer.activate()
                mode = SystemMode.HOME
            elif mode != SystemMode.HOME:
                # don't switch to moving home when already at home: this can be
                # triggered by small amounts of noise in the position estimate
                mode = SystemMode.MOVING_HOME

            # mode switch
            if mode != prev_mode:
                mode_start_time = t
                mode_t = 0
                print(f"mode = {mode}")
            prev_mode = mode

            # move based on mode
            base_vel_des = np.zeros(3)
            arm_cmd_vel = np.zeros(6)

            if mode == SystemMode.HOME:
                arm_q_err = home[3:] - q[3:]
                if USE_STABILIZER and home_stabilizer_timer.is_active(mode_t):
                    x = stabilizer.update(q, tray.position, dt)
                    if x is None:
                        print("failed to solve stabilizer QP")
                        arm_cmd_vel = np.zeros(6)
                    else:
                        arm_cmd_vel = x[3:]
                # TODO
                # elif np.linalg.norm(arm_q_err) > CONVERGENCE_TOL:
                #     # move arm back to home after stabilizing
                #     arm_cmd_vel = Kp * arm_q_err
            elif mode == SystemMode.SERVING:
                if USE_STABILIZER and serving_stabilizer_timer.is_active(mode_t):
                    x = stabilizer.update(q, tray.position, dt)
                    if x is None:
                        print("failed to solve stabilizer QP")
                        arm_cmd_vel = np.zeros(6)
                    else:
                        arm_cmd_vel = x[3:]
            elif mode == SystemMode.MOVING_HOME:
                base_error = home[:3] - q[:3]
                # base_error[2] = mm.wrap_to_pi(base_error[2])
                vd = Kp * base_error

                # rotate into the body frame
                C_bw = rotz(-q[2])
                vd = C_bw @ vd

                # keep current angle
                base_vel_des = np.append(vd[:2], 0)

            elif mode == SystemMode.FOLLOWING_TARGET:
                # base motion
                ang_err = node.compute_angular_error()
                vx = BASE_VEL_MAX[0] * (1 - np.abs(ang_err))
                base_vel_des = np.array([vx, 0, Kω * ang_err])

                # arm motion
                height_err = node.compute_height_error()
                vz_des = Kz * height_err
                arm_cmd_vel = servo_arm_up(q[3:], vz_des, dt)
            else:
                raise ValueError(f"Invalid mode: {mode}")

            # collision avoidance
            base_vel_des = node.filter_safe_velocity(base_vel_des)

            # accelerate toward desired velocity
            base_cmd_vel = sd.change_velocity(
                v=base_cmd_vel, vd=base_vel_des, max_a=BASE_ACC_MAX, dt=dt
            )

            # enforce velocity limits
            # TODO should I enforce arm velocity limits as well?
            base_cmd_vel = limit_base_vel(base_cmd_vel)

            # build the full robot joint command
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
    finally:
        if not args.dry_run:
            robot.brake()


if __name__ == "__main__":
    main()
