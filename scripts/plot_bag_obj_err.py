#!/usr/bin/env python3
"""Plot transported object error over time (that is, the distance of the object
   from its initial position over time)."""
import argparse

import numpy as np
import rosbag
import matplotlib.pyplot as plt

import upright_core as core
from mobile_manipulation_central import ros_utils
from upright_ros_interface.parsing import parse_object_error

import IPython


TRAY_VICON_NAME = "ThingRoundTray"
# TRAY_VICON_NAME = "ThingWoodTray"


def get_bag_topics(bag):
    return list(bag.get_type_and_topic_info()[1].keys())


def vicon_object_topics(bag):
    topics = get_bag_topics(bag)

    def func(topic):
        if not topic.startswith("/vicon"):
            return False
        if (
            topic.endswith("markers")
            or "ThingBase" in topic
            or TRAY_VICON_NAME in topic
        ):
            return False
        return True

    topics = list(filter(func, topics))
    if len(topics) == 0:
        print("No object topic found!")
    elif len(topics) > 1:
        raise ValueError("Multiple object topics found!")
    return topics[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bagfile", help="Bag file to plot.")
    args = parser.parse_args()

    bag = rosbag.Bag(args.bagfile)
    object_vicon_name = vicon_object_topics(bag).split("/")[-1]
    print(f"Object is {object_vicon_name}")
    errors, ts = parse_object_error(
        bag, TRAY_VICON_NAME, object_vicon_name, return_times=True
    )

    plt.figure()
    plt.plot(ts, 1000 * errors)  # convert to mm
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Distance error [mm]")

    plt.show()


if __name__ == "__main__":
    main()
