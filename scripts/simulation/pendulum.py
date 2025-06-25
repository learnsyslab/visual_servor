import cvxpy as cp
import numpy as np
import rigeo as rg
from scipy.linalg import expm
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

import IPython

np.set_printoptions(precision=4, suppress=True)


ACCEL = np.array([2, 0, 0])
DURATION = 10
STEPS = 1000
TIMESTEP = DURATION / STEPS

GRAVITY = np.array([0, 0, -9.81])

# length of the pendulum
LENGTH = 0.3
DAMPING = 0

# height of the object's CoM above its base
OBJ_COM_HEIGHT = 0.0
OBJ_BASE_HALF_LENGTH = 0.1

TRAY_RADIUS = 0.2

# object-tray friction coefficient
MU = 0.5


def input_accel(t):
    if t <= 1:
        return ACCEL
    elif t <= 2:
        return -ACCEL
    return np.zeros_like(ACCEL)


def unit(v):
    """Normalize to a unit vector."""
    norm = np.linalg.norm(v)
    if np.isclose(norm, 0):
        return np.zeros_like(v)
    return v / norm


def contact_jac(point):
    return np.vstack((rg.skew3(point), np.eye(3)))


# inertial parameters
com_p = np.array([0, 0, -LENGTH])
com_o = com_p + np.array([0, 0, OBJ_COM_HEIGHT])

# NOTE: inertias are really important for all results!
params_p = rg.InertialParameters(
    mass=1,
    com=com_p,
    I=np.diag([0.001, 0.001, 0.001]),
    translate_from_com=True,
)
params_o = rg.InertialParameters(
    mass=1, com=com_o, I=0.0 * np.eye(3), translate_from_com=True
)
params_po = params_p + params_o

# spatial mass matrices
M_o = params_o.M
M_p = params_p.M

# contact points
contacts = np.array(
    [
        [OBJ_BASE_HALF_LENGTH, OBJ_BASE_HALF_LENGTH, -LENGTH],
        [OBJ_BASE_HALF_LENGTH, -OBJ_BASE_HALF_LENGTH, -LENGTH],
        [-OBJ_BASE_HALF_LENGTH, -OBJ_BASE_HALF_LENGTH, -LENGTH],
        [-OBJ_BASE_HALF_LENGTH, OBJ_BASE_HALF_LENGTH, -LENGTH],
    ]
)
Gs = [contact_jac(c) for c in contacts]

mG_o = params_o.mass * contact_jac(params_o.com)
mG_p = params_p.mass * contact_jac(params_p.com)

# anchor points for tray ropes
anchors = np.array(
    [
        [TRAY_RADIUS, TRAY_RADIUS, -LENGTH],
        [TRAY_RADIUS, -TRAY_RADIUS, -LENGTH],
        [-TRAY_RADIUS, -TRAY_RADIUS, -LENGTH],
        [-TRAY_RADIUS, TRAY_RADIUS, -LENGTH],
    ]
)


# friction matrix: F @ f <= 0 means f inside friction cone
F = np.array([[0, 0, -1], [1, 1, -MU], [1, -1, -MU], [-1, -1, -MU], [-1, 1, -MU]])

# initial state
ξ = rg.SV.zero()
C = np.eye(3)  # this is C_wb

# spatial acceleration
# we can compute the linear part analytically
ωdot = cp.Parameter(3)
vdot = cp.Parameter(3)
ξdot = cp.hstack([ωdot, vdot])

# remainder wrenches
rem_p = cp.Parameter(6)
rem_o = cp.Parameter(6)

# contact forces
fs = cp.Variable((4, 3))

# total contact wrench on the object
# this is applied the opposite way on the tray
wc = cp.sum([G @ f for G, f in zip(Gs, fs)])

# tensile force along ropes
# no torque because forces all act through the origin
ft = cp.Variable(3)
wt = cp.hstack((np.zeros(3), ft))

# minimize the shear contact forces
W = np.diag([1, 1, 0])
objective = cp.Minimize(cp.sum([cp.quad_form(f, W, assume_PSD=True) for f in fs]))

# Newton-Euler equations (force-torque balance)
constraints = [rem_o + wc == M_o @ ξdot, rem_p - wc + wt == M_p @ ξdot]

# friction constraints
constraints.extend([F @ f <= 0 for f in fs])

# tensile forces must be positive
fts = cp.Variable(4, nonneg=True)
constraints.append(ft == cp.sum([-f * unit(a) for f, a in zip(fts, anchors)]))

problem = cp.Problem(objective, constraints)

ts = []
us = []
φs = []
ξs = []
ffs = []  # TODO name

t = 0
for i in range(STEPS):
    t = i * TIMESTEP
    u = input_accel(t)

    # solve for spatial acceleration
    ωdot.value = np.linalg.solve(
        params_po.I,
        np.cross(params_po.h, C.T @ (GRAVITY - u))
        - np.cross(ξ.angular, params_po.I @ ξ.angular),
    )
    vdot.value = C.T @ u - np.cross(ξ.angular, ξ.linear)

    # gravity and Coriolis forces on each body
    rem_p.value = ξ.adjoint().T @ M_p @ ξ.vec + mG_p @ C.T @ GRAVITY
    rem_o.value = ξ.adjoint().T @ M_o @ ξ.vec + mG_o @ C.T @ GRAVITY

    problem.solve(solver=cp.CLARABEL)
    if problem.status != "optimal":
        print(f"failed to solve at time {t}")
        IPython.embed()
        break

    # integrate forward in time
    ξ = ξ + rg.SV(linear=vdot.value, angular=ωdot.value) * TIMESTEP
    C = C @ expm(rg.skew3(ξ.angular) * TIMESTEP)

    ts.append(t)
    us.append(u)
    φs.append(Rotation.from_matrix(C).as_rotvec())

    # rotate into the world frame
    ξs.append(np.concatenate((C @ ξ.angular, C @ ξ.linear)))

    ffs.append(fs.value)

us = np.array(us)
φs = np.array(φs)
ξs = np.array(ξs)
ffs = np.array(ffs)

# φs[φs < 0] += 2 * np.pi

plt.figure()
plt.title("Input Acceleration")
plt.plot(ts, us[:, 0], label="x")
plt.plot(ts, us[:, 1], label="y")
plt.plot(ts, us[:, 2], label="z")
plt.grid()
plt.xlabel("Time [s]")
plt.ylabel("Acceleration [m/s^2]")
plt.legend()

plt.figure()
plt.title("Orientation")
plt.plot(ts, φs[:, 0], label="x")
plt.plot(ts, φs[:, 1], label="y")
plt.plot(ts, φs[:, 2], label="z")
plt.grid()
plt.xlabel("Time [s]")
plt.ylabel("Angle [rad]")
plt.legend()

plt.figure()
plt.title("Linear velocity")
plt.plot(ts, ξs[:, 3], label="x")
plt.plot(ts, ξs[:, 4], label="y")
plt.plot(ts, ξs[:, 5], label="z")
plt.grid()
plt.xlabel("Time [s]")
plt.ylabel("Linear velocity [m/s]")
plt.legend()

plt.figure()
plt.title("Angular velocity")
plt.plot(ts, ξs[:, 0], label="x")
plt.plot(ts, ξs[:, 1], label="y")
plt.plot(ts, ξs[:, 2], label="z")
plt.grid()
plt.xlabel("Time [s]")
plt.ylabel("Angular velocity [rad/s]")
plt.legend()

plt.figure()
plt.title("Contact forces")
plt.plot(ts, ffs[:, 0, 0], label="x")
plt.plot(ts, ffs[:, 0, 1], label="y")
plt.plot(ts, ffs[:, 0, 2], label="z")
plt.grid()
plt.xlabel("Time [s]")
plt.ylabel("Contact force [N]")
plt.legend()

plt.show()
