#!/usr/bin/env python3
"""Stabilize spatial pendulum using LQR."""
import argparse
from pathlib import Path

import numpy as np
import rospy
import rospkg
import yaml

import mobile_manipulation_central as mm
import visual_servor as vs

RATE = 125
ACCEL_MAX = 0.5
VEL_MAX = 0.1
JOINT_VEL_MAX = 0.1
USE_INTEGRAL_TERM = True


def main():
    np.set_printoptions(suppress=True, precision=5)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run", action="store_true", help="Don't send commands to the robot."
    )
    parser.add_argument(
        "--base", action="store_true", help="Move the mobile base rather than the arm."
    )
    args = parser.parse_args()

    # load pendulum calibration
    r_te_e = np.array(
        mm.load_pkg_config(
            pkg_name="visual_servor", relpath="config/pendulum_calibration.yaml"
        )["r_te_e"]
    )

    rospy.init_node("pendulum_node", disable_signals=True)

    rate = rospy.Rate(RATE)
    dt = 1.0 / RATE

    robot = mm.MobileManipulatorROSInterface()
    tray = mm.ViconObjectInterface("ThingRoundTray")
    signal_handler = mm.RobotSignalHandler(robot, args.dry_run)

    model = mm.MobileManipulatorKinematics(tool_link_name="ur10_arm_tool0")
    stabilizer = vs.PendulumStabilizer(
        model=model,
        vel_max=VEL_MAX,
        accel_max=ACCEL_MAX,
        joint_vel_max=JOINT_VEL_MAX,
        use_integral_term=USE_INTEGRAL_TERM,
    )

    print("Waiting for robot...")
    while not rospy.is_shutdown() and not (robot.ready() and tray.ready()):
        rate.sleep()
    print("...robot ready.")

    stabilizer.init(robot.q, r_te_e)

    try:
        t0 = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():
            t = rospy.Time.now().to_sec() - t0

            cmd_vel = stabilizer.update(robot.q, tray.position, dt, base=args.base)
            if args.base:
                cmd_vel = np.concatenate((cmd_vel, np.zeros(6)))
            else:
                cmd_vel = np.concatenate((np.zeros(3), cmd_vel))

            # send command to robot
            if args.dry_run:
                print(f"cmd_vel = {cmd_vel}")
            else:
                robot.publish_cmd_vel(cmd_vel, bodyframe=False)

            rate.sleep()
    finally:
        if not args.dry_run:
            robot.brake()


if __name__ == "__main__":
    main()
