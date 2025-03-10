"""Visualize difference ways to threshold range of obstacles."""
import numpy as np
import matplotlib.pyplot as plt


ANGLE_LIMIT = np.pi / 4

SIDE_LIMIT = 0.5
FRONT_LIMIT = 1

angles = np.linspace(-ANGLE_LIMIT, ANGLE_LIMIT, 1000)
ranges1 = np.ones_like(angles)

a = (FRONT_LIMIT - SIDE_LIMIT) / ANGLE_LIMIT
ranges2 = a * (ANGLE_LIMIT - np.abs(angles)) + SIDE_LIMIT

a = (FRONT_LIMIT - SIDE_LIMIT) / ANGLE_LIMIT**2
ranges3 = a * (ANGLE_LIMIT - np.abs(angles)) ** 2 + SIDE_LIMIT

a = (SIDE_LIMIT - FRONT_LIMIT) / (np.cos(ANGLE_LIMIT) - 1)
b = FRONT_LIMIT - a
ranges4 = a * np.cos(angles) + b

plt.figure()
plt.plot([0], [0], "o", color="k")
plt.plot(angles, ranges1)
plt.plot(angles, ranges2)
plt.plot(angles, ranges3)
plt.plot(angles, ranges4)
plt.grid()
plt.xlabel("Angle [rad]")
plt.ylabel("Ranges [m]")

xy1 = np.array([ranges1 * np.cos(angles), ranges1 * np.sin(angles)])
xy2 = np.array([ranges2 * np.cos(angles), ranges2 * np.sin(angles)])
xy3 = np.array([ranges3 * np.cos(angles), ranges3 * np.sin(angles)])
xy4 = np.array([ranges4 * np.cos(angles), ranges4 * np.sin(angles)])

plt.figure()
plt.plot([0], [0], "o", color="k")
plt.plot(xy1[0, :], xy1[1, :])
plt.plot(xy2[0, :], xy2[1, :])
plt.plot(xy3[0, :], xy3[1, :])
plt.plot(xy4[0, :], xy4[1, :])
plt.grid()
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.axis("equal")

plt.show()
