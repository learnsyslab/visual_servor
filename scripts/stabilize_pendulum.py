#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import rospy
import rospkg
import yaml

import mobile_manipulation_central as mm
import serving_demo as sd

RATE = 125
ACCEL_MAX = 0.5
VEL_MAX = 0.1
JOINT_VEL_MAX = 0.1


def main():
    np.set_printoptions(suppress=True, precision=5)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run", action="store_true", help="Don't send commands to the robot."
    )
    args = parser.parse_args()

    # load pendulum calibration
    rospack = rospkg.RosPack()
    pkg_path = Path(rospack.get_path("serving_demo"))
    calib_path = pkg_path / "config" / "pendulum_calibration.yaml"
    with open(calib_path) as f:
        calib = yaml.safe_load(f)
        r_tray_ee = np.array(calib["r_tray_ee"])

    rospy.init_node("pendulum_node", disable_signals=True)

    rate = rospy.Rate(RATE)
    dt = 1.0 / RATE

    robot = mm.MobileManipulatorROSInterface()
    tray = mm.ViconObjectInterface("ThingRoundTray")
    signal_handler = mm.RobotSignalHandler(robot, args.dry_run)

    model = mm.MobileManipulatorKinematics(tool_link_name="pendulum_pivot")
    stabilizer = sd.PendulumStabilizer(model=model)

    print("Waiting for robot...")
    while not rospy.is_shutdown() and not (robot.ready() and tray.ready()):
        rate.sleep()
    print("...robot ready.")

    stabilizer.init(robot.q, r_tray_ee)

    try:
        t0 = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():
            t = rospy.Time.now().to_sec() - t0
            q = robot.q

            cmd_vel = stabilizer.update(q, tray.position, dt)

            # send command to robot
            if args.dry_run:
                print(f"cmd_vel = {cmd_vel}")
            else:
                robot.publish_cmd_vel(cmd_vel, bodyframe=True)

            rate.sleep()
    finally:
        if not args.dry_run:
            robot.brake()


if __name__ == "__main__":
    main()
