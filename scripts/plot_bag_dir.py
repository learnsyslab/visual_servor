#!/usr/bin/env python3
"""Plot slider position from a ROS bag."""
import argparse
import glob
from pathlib import Path

import numpy as np
import rosbag
import matplotlib.pyplot as plt

import mobile_manipulation_central as mm
from mobile_manipulation_central import ros_utils


def parse_bag_dir(directory):
    """Parse bag path from a data directory.

    Returns bag path as a string."""
    dir_path = Path(directory)

    bag_files = glob.glob(dir_path.as_posix() + "/*.bag")
    if len(bag_files) == 0:
        raise FileNotFoundError(
            "Error: could not find a bag file in the specified directory."
        )
    if len(bag_files) > 1:
        raise FileNotFoundError(
            "Error: multiple bag files in the specified directory. Please specify the name using the `--bag_name` option."
        )
    bag_path = bag_files[0]
    return bag_path


def parse_state_msgs(msgs):
    ts = np.array([ros_utils.msg_time(msg) for msg in msgs])
    qds = np.array([msg.qd for msg in msgs])
    qs = np.array([msg.q for msg in msgs])
    cmd_vels = np.array([msg.cmd_vel for msg in msgs])
    return ts, qds, qs, cmd_vels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "bagdir", help="Directory containing bag file and pickled parameters."
    )
    args = parser.parse_args()

    bag_path = parse_bag_dir(args.bagdir)
    bag = rosbag.Bag(bag_path)

    state_msgs = [msg for _, msg, _ in bag.read_messages("/serving/state")]
    ts, qds, qs, cmd_vels = parse_state_msgs(state_msgs)

    # normalize time
    t0 = ts[0]
    ts -= t0

    plt.figure()
    plt.plot(ts, qs[:, 0], label="x")
    plt.plot(ts, qs[:, 1], label="y")
    plt.plot(ts, qs[:, 2], label="yaw")
    plt.plot(ts, qds[:, 0], label="xd")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.title("Base pose vs. time")
    plt.legend()
    plt.grid()

    plt.figure()
    plt.plot(ts, cmd_vels[:, 0], label="vx")
    plt.plot(ts, cmd_vels[:, 1], label="vy")
    plt.plot(ts, cmd_vels[:, 2], label="ω")
    plt.xlabel("Time [s]")
    plt.ylabel("Command")
    plt.title("Base commands vs. time")
    plt.legend()
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()
