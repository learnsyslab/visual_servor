import numpy as np
from scipy.linalg import expm, null_space, solve_continuous_are
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import rigeo as rg

import IPython


def lqr(A, B, Q, R):
    P = solve_continuous_are(A, B, Q, R)
    return np.linalg.solve(R, B.T @ P)


MASS = 1
LENGTH = 0.3
# INERTIA_COM = np.diag([0.01, 0.02, 0.03])
INERTIA_COM = 0.1 * np.array([[1, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 1]])
assert np.all(np.linalg.eigvals(INERTIA_COM) >= 0)
GRAVITY = -9.81

TIMESTEP = 0.001
DURATION = 10

# parameters
m = MASS
c = np.array([0, 0, -LENGTH])
I = INERTIA_COM - m * rg.skew3(c) @ rg.skew3(c)
g = np.array([0, 0, GRAVITY])
ρ = c / LENGTH
α = 0.1  # damping about ρ

# desired state
r = np.zeros(3)
v = np.zeros(3)
ω = np.zeros(3)
R = np.eye(3)

ρdot = np.cross(ω, ρ)

# x-y components of each of r, ρ, rdot, ρdot
A = np.zeros((8, 8))
A[0:2, 4:6] = np.eye(2)
A[2:4, 6:8] = np.eye(2)
A[6:8, 2:4] = ((rg.skew3(np.cross(ρ, g)) + rg.skew3(ρ) @ rg.skew3(g)) / LENGTH)[:2, :2]

B = np.zeros((8, 2))
B[4:6, :] = np.eye(2)
B[6:8, :] = (rg.skew3(ρ) @ rg.skew3(ρ) / LENGTH)[:2, :2]

# solve for feedback gain u = -K @ x with LQR
K = lqr(A, B, Q=np.eye(8), R=0.1 * np.eye(2))

# R = Rotation.from_rotvec([0.4*np.pi, 0, 0]).as_matrix()
# TODO we cannot have any angular velocity about z
ω = np.array([0.5, 0.3, 0])
# v = np.cross(ω, c)

rd = r
pd = rd - c

ts = []
rs = []
vs = []
ωs = []
φs = []

t = 0
i = 0
while t < DURATION:
    p = r - R @ c
    pdot = R @ (v - np.cross(ω, c))
    Δp = pd - p
    Δr = rd - r

    x = np.concatenate((r[:2], (R @ ρ)[:2], (R @ v)[:2], (R @ np.cross(ω, ρ))[:2]))
    u = np.zeros(3)
    u[:2] = -K @ x

    τ = m * np.cross(c, R.T @ (g - u))
    ωdot = np.linalg.solve(I, τ - np.cross(ω, I @ ω) - α * np.outer(ρ, ρ) @ ω)

    # position
    vdot = R.T @ u - np.cross(ω, v)
    v = v + vdot * TIMESTEP

    rdot = R @ v
    r = r + rdot * TIMESTEP

    # orientation
    ω = ω + ωdot * TIMESTEP
    R = R @ expm(rg.skew3(ω) * TIMESTEP)

    ts.append(t)
    rs.append(r)
    vs.append(v)
    ωs.append(ω)
    φs.append(Rotation.from_matrix(R).as_rotvec())

    i += 1
    t = i * TIMESTEP

ts = np.array(ts)
rs = np.array(rs)
vs = np.array(vs)
ωs = np.array(ωs)
φs = np.array(φs)

plt.figure()
plt.title("EE position")
plt.plot(ts, rs[:, 0], label="x")
plt.plot(ts, rs[:, 1], label="y")
plt.plot(ts, rs[:, 2], label="z")
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Position [m]")
plt.grid()

plt.figure()
plt.title("EE velocity")
plt.plot(ts, vs[:, 0], label="vx")
plt.plot(ts, vs[:, 1], label="vy")
plt.plot(ts, vs[:, 2], label="vz")
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("v [m/s]")
plt.grid()

plt.figure()
plt.title("Tray orientation")
plt.plot(ts, φs[:, 0], label="φx")
plt.plot(ts, φs[:, 1], label="φy")
plt.plot(ts, φs[:, 2], label="φz")
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("φ [rad]")
plt.grid()

plt.figure()
plt.title("Tray angular velocity")
plt.plot(ts, ωs[:, 0], label="ωx")
plt.plot(ts, ωs[:, 1], label="ωy")
plt.plot(ts, ωs[:, 2], label="ωz")
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("ω [rad/s]")
plt.grid()

plt.show()
