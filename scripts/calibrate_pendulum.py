#!/usr/bin/env python3
import numpy as np
import rospy
import yaml

from spatialmath.base import rotz

import mobile_manipulation_central as mm


RATE = 10
NUM_MEASUREMENTS = 10


def main():
    rospy.init_node("pendulum_calibrate_node")

    robot = mm.MobileManipulatorROSInterface()
    tray = mm.ViconObjectInterface("ThingRoundTray")
    model = mm.MobileManipulatorKinematics(tool_link_name="ur10_arm_tool0")

    rate = rospy.Rate(RATE)

    print("Waiting for robot...")
    while not rospy.is_shutdown() and not (robot.ready() and tray.ready()):
        rate.sleep()
    print("...robot ready.")

    # gather measurements
    r_te_es = []
    while not rospy.is_shutdown():
        model.forward(robot.q)
        r_ew_w = model.link_pose()[0]
        C_we = rotz(robot.q[2])
        r_tw_w = tray.position
        r_te_e = C_we.T @ (r_tw_w - r_ew_w)
        r_te_es.append(r_te_e)
        if len(r_te_es) >= NUM_MEASUREMENTS:
            break
        rate.sleep()

    # average measurements
    r_te_e = np.mean(r_te_es, axis=0).tolist()
    print(f"r_te_e = {r_te_e}")

    filename = "pendulum_calibration.yaml"
    with open(filename, "w") as f:
        yaml.dump({"r_te_e": r_te_e}, f)
    print(f"Wrote calibration to {filename}.")


if __name__ == "__main__":
    main()
