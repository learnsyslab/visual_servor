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

MIN_ANGLE = -np.pi / 4.0
MAX_ANGLE = np.pi / 4.0

VEL_DAMP_COEFF = 1
VEL_DAMP_SAFETY = 0
VEL_DAMP_INFL = 0.5


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

    def _teleop_cb(self, msg):
        # desired base velocity
        cmd_vel_des = np.array([msg.linear.x, msg.linear.y, msg.angular.z])
        self.cmd_vel_des = np.clip(cmd_vel_des, -0.1, 0.1)

    def _scan_cb(self, scan):
        """Get ranges and angles from a scan."""

        # TODO tune this
        lidar_position = np.array([0.25, 0])

        # construct the raw points
        n = len(scan.ranges)
        ranges = np.array(scan.ranges)
        angles = np.array([scan.angle_min + i * scan.angle_increment for i in range(n)])
        points = (np.vstack((np.cos(angles), np.sin(angles))) * ranges).T

        # remove points at invalid angles
        # valid = (angles >= MIN_ANGLE) & (angles <= MAX_ANGLE)
        points = points[ranges >= scan.range_min, :]

        # relative to the base reference frame
        self.points = points + lidar_position

    def filter_safe_velocity(self, lin_vel, ang_vel):
        if len(self.points) == 0:
            return lin_vel, ang_vel

        # define bounding ellipsoid
        rx = 0.75
        ry = 0.5
        A = np.diag([1.0 / rx**2, 1.0 / ry**2])
        c = np.array([0.25, 0])

        A_safe = np.diag([1.0 / rx**2, 1.0 / ry**2])
        A_infl = np.diag(
            [1.0 / (rx + VEL_DAMP_INFL) ** 2, 1.0 / (ry + VEL_DAMP_INFL) ** 2]
        )

        # TODO now I need to compute points within influence dist of ellipsoid

        # remove points outside of the influence ellipse
        points = self.points
        x = points - c
        # TODO these need to be normalized to be 1?
        normals = x @ A_infl
        valid = np.sum(x * normals, axis=1) <= 1

        x = x[valid, :]
        points = points[valid, :]
        normals = normals[valid, :]

        n = len(points)
        # print(np.min(np.linalg.norm(x, axis=1)))
        # print(f"num points = {n}")
        if n == 0:
            # none of the points are inside the ellipse
            return lin_vel, ang_vel

        # distances to safety ellipsoid
        # TODO check that this is right
        normals_safe = x @ A_safe
        denoms = np.sqrt(np.sum(x * normals_safe, axis=1))
        closests = normals_safe / denoms[:, None]

        # actually normalize
        normals = normals / np.linalg.norm(normals, axis=1)[:, None]

        dists = np.sum(normals * (x - closests), axis=1)

        P = np.eye(3)
        ξd = np.append(lin_vel, ang_vel)
        # h = np.zeros(n)
        h = VEL_DAMP_COEFF * dists / VEL_DAMP_INFL

        print(f"min dist = {np.min(dists)}")
        IPython.embed()
        return
        # return lin_vel, ang_vel
        # IPython.embed()
        # return

        # TODO we can bucket the points to avoid so many constraints
        # TODO velocity damper formulation?
        S = np.array([[0, -1], [1, 0]])
        zs = np.sum(normals * (S @ points.T).T, axis=1)
        G = np.hstack((normals, zs[:, None]))

        x = solve_qp(P=P, q=-ξd, G=G, h=h, solver="quadprog")
        if x is None:
            print("failed to solve obstacle avoidance QP")
            return np.zeros(2), 0
        return x[:2], x[2]


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
            print(f"desired = {cmd_vel_des}")
            print(f"actual  = {cmd_vel}")

        robot.publish_cmd_vel(cmd_vel, bodyframe=True)
        rate.sleep()

    robot.brake()


if __name__ == "__main__":
    main()
