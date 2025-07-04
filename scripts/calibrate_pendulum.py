import numpy as np
import rospy
import yaml

import mobile_manipulation_central as mm


RATE = 10
NUM_MEASUREMENTS = 10


def main():
    rospy.init_node("pendulum_calibrate_node")

    robot = mm.MobileManipulatorROSInterface()
    tray = mm.ViconObjectInterface("ThingRoundTray")
    model = mm.MobileManipulatorKinematics(tool_link_name="pendulum_pivot")

    rate = rospy.Rate(RATE)

    print("Waiting for robot...")
    while not rospy.is_shutdown() and not (robot.ready() and tray.ready()):
        rate.sleep()
    print("...robot ready.")

    # gather measurements
    r_tray_ees = []
    while not rospy.is_shutdown():
        model.forward(robot.q)
        r_ee = model.link_pose()[0]
        r_tray = tray.position
        r_tray_ees.append(r_tray - r_ee)
        if len(r_tray_ees) >= NUM_MEASUREMENTS:
            break
        rate.sleep()

    # average measurements
    r_tray_ee = np.mean(r_tray_ees, axis=0).tolist()
    print(f"r_tray_ee = {r_tray_ee}")

    filename = "pendulum_calibration.yaml"
    with open(filename, "w") as f:
        yaml.dump({"r_tray_ee": r_tray_ee}, f)
    print(f"Wrote calibration to {filename}.")


if __name__ == "__main__":
    main()
