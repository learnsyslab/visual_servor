#!/usr/bin/env python3
import numpy as np
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from qpsolvers import solve_qp

import mobile_manipulation_central as mm
import serving_demo as sd

import IPython

RATE = 100

LIDAR_OFFSET = np.array([0.25, 0])
VEL_MAX = 0.1


class CollisionNode:
    def __init__(self):
        self.points = []
        self.cmd_vel_des = np.zeros(3)

        self.scan_sub = rospy.Subscriber(
            "/front/scan", LaserScan, self._scan_cb, queue_size=1
        )
        self.teleop_sub = rospy.Subscriber(
            "/teleop_cmd_vel",
            Twist,
            self._teleop_cb,
            queue_size=1,
        )

        self.collision = sd.CollisionEllipse(rx=0.8, ry=0.6, center=[0.25, 0])

    def _teleop_cb(self, msg):
        # desired base velocity
        cmd_vel_des = np.array([msg.linear.x, msg.linear.y, msg.angular.z])
        self.cmd_vel_des = np.clip(cmd_vel_des, -VEL_MAX, VEL_MAX)

    def _scan_cb(self, scan):
        """Get ranges and angles from a scan."""

        # construct the raw points
        n = len(scan.ranges)
        ranges = np.array(scan.ranges)
        angles = np.array([scan.angle_min + i * scan.angle_increment for i in range(n)])
        points = (np.vstack((np.cos(angles), np.sin(angles))) * ranges).T

        # remove invalid points
        valid = (ranges >= scan.range_min) & (ranges <= scan.range_max)
        points = points[valid, :]

        # relative to the base reference frame
        self.points = points + LIDAR_OFFSET

    def filter_safe_velocity(self, lin_vel, ang_vel):
        return self.collision.filter_safe_velocity(lin_vel, ang_vel, self.points)


def main():
    rospy.init_node("collision_node", disable_signals=True)
    node = CollisionNode()

    rate = rospy.Rate(RATE)

    robot = mm.RidgebackROSInterface()
    signal_handler = mm.RobotSignalHandler(robot)

    # wait until robot feedback has been received
    print("Waiting for robot...")
    while not rospy.is_shutdown() and not robot.ready():
        rate.sleep()
    print("...robot ready.")

    while not rospy.is_shutdown():
        cmd_vel_des = node.cmd_vel_des
        lin_vel, ang_vel = node.filter_safe_velocity(cmd_vel_des[:2], cmd_vel_des[2])
        cmd_vel = np.append(lin_vel, ang_vel)

        if not np.allclose(cmd_vel_des, cmd_vel):
            print(f"diff = {cmd_vel_des - cmd_vel}")

        robot.publish_cmd_vel(cmd_vel, bodyframe=True)
        rate.sleep()

    robot.brake()


if __name__ == "__main__":
    main()
