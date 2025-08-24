#!/usr/bin/env python3
import numpy as np
import rospy
import yaml

from spatialmath.base import rotz, q2r

import mobile_manipulation_central as mm


RATE = 10
NUM_MEASUREMENTS = 10


def main():
    rospy.init_node("base_pendulum_node")

    tray = mm.ViconObjectInterface("ThingRoundTray")
    base = mm.ViconObjectInterface("ThingBase_4")

    q0 = np.array([-1.0, 0, 0, 1.5708, -1.5708, 1.5708, 0,  1.5708, -0.2618])
    model = mm.MobileManipulatorKinematics(tool_link_name="ur10_arm_tool0")
    model.forward(q0)
    print(model.link_pose()[0])

    q = q0.copy()
    q[0] += 3
    model.forward(q)
    print(model.link_pose()[0])
    return

    rate = rospy.Rate(RATE)

    print("Waiting for Vicon...")
    while not rospy.is_shutdown() and not (tray.ready() and base.ready()):
        rate.sleep()
    print("...Vicon ready.")

    while not rospy.is_shutdown():
        r_bw_w = base.position
        C_wb = q2r(base.orientation, order="xyzs")
        r_tw_w = tray.position

        r_tb_b = C_wb.T @ (r_tw_w - r_bw_w)
        print(f"r_tb_b = {r_tb_b}")

        rate.sleep()


# r_tb_b = [0.70787938 0.19991785 0.7462641 ]


if __name__ == "__main__":
    main()
