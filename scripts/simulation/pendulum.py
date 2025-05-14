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

ALLOW_ROTATION = True
USE_FRICTION_CONSTRAINTS = True
USE_TENSILE_FORCE_CONSTRAINT = True

# height of the object's CoM above its base
OBJ_COM_HEIGHT = 0.05
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


# inertial parameters
com_p = np.array([0, 0, -LENGTH])
com_o = com_p + np.array([0, 0, OBJ_COM_HEIGHT])

params_p = rg.InertialParameters(
    mass=1,
    com=com_p,
    I=2 * np.diag([0.01, 0.02, 0.03]),
    translate_from_com=True,
)
params_o = rg.InertialParameters(
    mass=1, com=com_o, I=0.01 * np.eye(3), translate_from_com=True
)
params_po = params_p + params_o

# spatial mass matrices
M_o = params_o.M
M_p = params_p.M
M_po = params_po.M


def contact_jac(point):
    return np.vstack((rg.skew3(point), np.eye(3)))


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
mG_po = params_po.mass * contact_jac(params_po.com)

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
F = np.array(
    [[0, 0, -1], [1, 1, -MU], [1, -1, -MU], [-1, -1, -MU], [-1, 1, -MU]]
)

# initial state
ξ = rg.SV.zero()
C = np.eye(3)  # this is C_wb


# spatial acceleration
# we can compute the linear part analytically
ωdot = cp.Variable(3)
vdot = cp.Parameter(3)
ξdot = cp.hstack([ωdot, vdot])

# remainder wrenches
rem_po = cp.Parameter(6)
rem_o = cp.Parameter(6)

# contact forces
fs = cp.Variable((4, 3))
z = fs.flatten()

# tensile force along c
ft = cp.Variable(3)
wt = cp.hstack((np.zeros(3), ft))

# total contact wrench on the object
wc = cp.sum([G @ f for G, f in zip(Gs, fs)])

# minimize the contact forces
objective = cp.Minimize(cp.quad_form(z, np.eye(12), assume_PSD=True))

constraints = [rem_o + wc == M_o @ ξdot]

if USE_FRICTION_CONSTRAINTS:
    constraints.extend([F @ f <= 0 for f in fs])

if USE_TENSILE_FORCE_CONSTRAINT:
    # also enforce that tensile forces must be positive
    # constraints.append(ft @ params_po.com <= 0)
    fts = cp.Variable(4, nonneg=True)
    constraints.append(ft == cp.sum([-f * unit(a) for f, a in zip(fts, anchors)]))

if ALLOW_ROTATION:
    constraints.append(rem_po + wt == M_po @ ξdot)
else:
    constraints.append(ωdot == 0)

problem = cp.Problem(objective, constraints)

ts = []
us = []
φs = []
ξs = []

# TODO why is there z velocity?
t = 0
for i in range(STEPS):
    t = i * TIMESTEP
    u = input_accel(t)

    # manually damp out the system
    # if t > 2:
    #     u = C @ (1 * np.cross(ξ.angular, params_p.com) - 1 * ξ.linear + np.cross(ξ.angular, ξ.linear))

    # inject energy into the system and then stabilize velocity
    if 2 < t < 4:
        # TODO it is unlikely we can actually pull this off with only a tensile force
        vpx = (C @ np.cross(ξ.angular, params_p.com))[0]
        u = np.array([-vpx, 0, 0])
    elif t >= 4:
        u = -C @ ξ.linear

    vdot.value = C.T @ u - np.cross(ξ.angular, ξ.linear)

    wg_po = mG_po @ C.T @ GRAVITY
    rem_po.value = ξ.adjoint().T @ M_po @ ξ.vec + wg_po
    # + np.concatenate((-DAMPING * ξ.angular, np.zeros(3)))

    wg_o = mG_o @ C.T @ GRAVITY
    rem_o.value = ξ.adjoint().T @ M_o @ ξ.vec + wg_o

    problem.solve(solver=cp.CLARABEL)
    if problem.status != "optimal":
        print(f"failed to solve at time {t}")
        IPython.embed()
        break

    ξ = ξ + rg.SV(linear=vdot.value, angular=ωdot.value) * TIMESTEP
    C = C @ expm(rg.skew3(ξ.angular) * TIMESTEP)

    ts.append(t)
    us.append(u)
    φs.append(Rotation.from_matrix(C).as_rotvec())

    # rotate into the world frame
    ξs.append(np.concatenate((C @ ξ.angular, C @ ξ.linear)))

us = np.array(us)
φs = np.array(φs)
ξs = np.array(ξs)

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

plt.show()
