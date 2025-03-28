#!/usr/bin/env python3
import argparse

import numpy as np
import rospy
from qpsolvers import solve_qp

import mobile_manipulation_central as mm

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

    rospy.init_node("pendulum_node", disable_signals=True)

    rate = rospy.Rate(RATE)
    dt = 1.0 / RATE
    α = 0.5

    robot = mm.MobileManipulatorROSInterface()
    tray = mm.ViconObjectInterface("ThingRoundTray")
    signal_handler = mm.RobotSignalHandler(robot, args.dry_run)

    # TODO check that this is the correct tool link
    model = mm.MobileManipulatorKinematics(tool_link_name="ur10_arm_tool0")

    tray_vel_filter = mm.ExponentialSmoother(τ=0.01)

    print("Waiting for robot...")
    while not rospy.is_shutdown() and not (robot.ready() and tray.ready()):
        rate.sleep()
    print("...robot ready.")

    # QP parameters
    # opt variables: arm_vel, s
    P = np.diag(np.append(np.ones(6), 0.01))
    qq = np.append(np.zeros(6), -1)
    ub = np.append(JOINT_VEL_MAX * np.ones(6), 1)
    lb = np.append(-JOINT_VEL_MAX * np.ones(6), 0)
    b = np.zeros(6)

    tray_pos_prev = tray.position

    v_ee = np.zeros(3)
    t0 = rospy.Time.now().to_sec()
    while not rospy.is_shutdown():
        t = rospy.Time.now().to_sec() - t0
        q = robot.q

        # model.forward(q)
        # r_ew_w, C_we = model.link_pose(rotation_matrix=True)

        # numerical diff and filter to estimate tray velocity
        tray_pos = tray.position
        v_tray_raw = (tray_pos - tray_pos_prev) / dt
        tray_pos_prev = tray_pos
        v_tray = tray_vel_filter.update(v_tray_raw, dt)

        # compute acceleration input
        # TODO somehow I would like to stabilize to a particular q as well?
        u = α * (v_tray - 2 * v_ee)
        u = np.clip(u, -ACCEL_MAX, ACCEL_MAX)

        # integrate to get commanded velocity
        # this is in the world frame
        v_ee += dt * u
        v_ee = np.clip(v_ee, -VEL_MAX, VEL_MAX)

        print(v_ee)

        # diff IK QP
        J = model.jacobian(q)
        # A = J[:, 3:]  # only arm
        # b = np.concatenate((v_ee, np.zeros(3)))
        ξ_ee = np.concatenate((v_ee, np.zeros(3)))
        A = np.hstack((J[:, 3:], -ξ_ee.reshape((6, 1))))

        # TODO use problem class?
        x = solve_qp(P=P, q=qq, A=A, b=b, lb=lb, ub=ub, solver="quadprog", verbose=True)
        if x is None:
            print("failed to solve QP")
            import IPython
            IPython.embed()
            break
        arm_cmd_vel = x[:6]
        cmd_vel = np.concatenate((np.zeros(3), arm_cmd_vel))

        # send command to robot
        if args.dry_run:
            print(f"v_tray = {v_tray}")
            print(f"q = {q}")
            print(f"cmd_vel = {cmd_vel}")
        else:
            robot.publish_cmd_vel(cmd_vel, bodyframe=True)

        rate.sleep()

    if not args.dry_run:
        robot.brake()


if __name__ == "__main__":
    main()
