#!/usr/bin/env python3
import argparse

import rospy
import numpy as np

import mobile_manipulation_central as mm
import serving_demo as sd
from serving_demo.msg import SystemState


RATE = 125  # Hz

CONVERGENCE_TOL = 1e-2

# durations of each segment
TRAJECTORY_DURATION = 4
STATIONARY_DURATION_1 = 1
STABILIZE_DURATION = 10
STATIONARY_DURATION_2 = 3


def main():
    np.set_printoptions(suppress=True, precision=6)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trajectory",
        choices=["slow", "fast"],
        required=True,
        help="The trajectory to use.",
    )
    parser.add_argument(
        "--tray",
        choices=["pendulum", "static"],
        required=True,
        help="The type of tray being used.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Don't send commands to the robot."
    )
    args = parser.parse_args()

    # load pendulum calibration
    r_te_e = np.array(
        mm.load_pkg_config(
            pkg_name="serving_demo", relpath="config/pendulum_calibration.yaml"
        )["r_te_e"]
    )

    rospy.init_node("pendulum_trial_node", disable_signals=True)

    state_pub = rospy.Publisher("/serving/state", SystemState, queue_size=1)

    rate = rospy.Rate(RATE)
    dt = 1.0 / RATE

    robot = mm.MobileManipulatorROSInterface()
    # TODO this will depend on tray type as well
    tray = mm.ViconObjectInterface("ThingRoundTray")
    signal_handler = mm.RobotSignalHandler(robot, args.dry_run)

    model = mm.MobileManipulatorKinematics(tool_link_name="ur10_arm_tool0")
    stabilizer = sd.PendulumStabilizer(model=model, use_integral_term=True)

    # wait until robot feedback has been received
    print("Waiting for robot...")
    while not rospy.is_shutdown() and not robot.ready():
        rate.sleep()
    print("...robot ready.")

    # wait a bit to make sure publishers are set up
    rospy.sleep(1.0)

    q0 = robot.q.copy()

    stabilizer.init(q0, r_te_e)
    stabilizing = False

    if args.trajectory == "slow":
        traj = sd.TrapezoidalTrajectory(a=0.5, t1=2, t2=2)
    elif args.trajectory == "fast":
        traj = sd.TrapezoidalTrajectory(a=1, t1=1, t2=3)
    else:
        raise ValueError("unknown trajectory")

    # position gain
    K = 1

    try:
        t0 = rospy.Time.now().to_sec()
        t = 0

        # first loop runs until base has completed its trajectory and converged
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            t = now.to_sec() - t0

            # base command
            qd, vd = traj.sample(t)
            base_vd = np.array([vd, 0, 0])
            base_qd = np.array([qd, 0, 0]) + q0[:3]
            base_err = base_qd - robot.q[:3]
            base_cmd_vel = K * base_err + base_vd

            arm_cmd_vel = np.zeros(6)

            base_converged = (
                np.linalg.norm(base_err) <= CONVERGENCE_TOL
                and np.linalg.norm(base_cmd_vel) <= CONVERGENCE_TOL
            )
            if t >= traj.duration and base_converged:
                break

            cmd_vel = np.concatenate((base_cmd_vel, arm_cmd_vel))

            # send command to the robot
            if not args.dry_run:
                robot.publish_cmd_vel(cmd_vel, bodyframe=False)
            # else:
            #     print(cmd_vel)

            msg = SystemState()
            msg.header.stamp = now
            msg.qd = list(np.concatenate((base_qd, q0[3:])))
            msg.q = list(robot.q)
            msg.cmd_vel = list(cmd_vel)
            state_pub.publish(msg)

            rate.sleep()

        robot.brake()
        print("base converged")
        t1 = t

        # second loop controls subsequent behaviour
        # for the pendulum we have to do stabilization, but with the static
        # tray we skip it
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            t = now.to_sec() - t0

            base_cmd_vel = np.zeros(3)
            arm_cmd_vel = np.zeros(6)

            if args.tray == "static":
                if t >= t1 + STATIONARY_DURATION_2:
                    break
            elif args.tray == "pendulum":
                if t <= t1 + STATIONARY_DURATION_1:
                    # wait a bit before stabilizing
                    pass
                elif t <= t1 + STATIONARY_DURATION_1 + STABILIZE_DURATION:
                    if not stabilizing:
                        stabilizer.reset(robot.q)
                        stabilizing = True
                        print("stabilizing")
                    else:
                        arm_cmd_vel = stabilizer.update(robot.q, tray.position, dt)
                else:
                    break
            else:
                raise ValueError("unknown tray type")

            cmd_vel = np.concatenate((base_cmd_vel, arm_cmd_vel))

            # send command to the robot
            if not args.dry_run:
                robot.publish_cmd_vel(cmd_vel, bodyframe=False)

            msg = SystemState()
            msg.header.stamp = now
            msg.qd = list(np.concatenate((base_qd, q0[3:])))
            msg.q = list(robot.q)
            msg.cmd_vel = list(cmd_vel)
            state_pub.publish(msg)

            rate.sleep()
    finally:
        if not args.dry_run:
            robot.brake()


if __name__ == "__main__":
    main()
