import numpy as np
from scipy.linalg import expm, null_space
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import rigeo as rg

import IPython


MASS = 1
LENGTH = 0.3
# INERTIA_COM = np.diag([0.01, 0.02, 0.03])
INERTIA_COM = 0.1 * np.array([[1, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 1]])
assert np.all(np.linalg.eigvals(INERTIA_COM) >= 0)
GRAVITY = -9.81

TIMESTEP = 0.001
DURATION = 20

# parameters
m = MASS
c = np.array([0, 0, -LENGTH])
I = INERTIA_COM - m * rg.skew3(c) @ rg.skew3(c)
g = np.array([0, 0, GRAVITY])
ρ = c / LENGTH
α = 0.1  # damping about ρ


def energy(R, ω):
    return 0.5 * ω.T @ I @ ω - m * (GRAVITY * LENGTH + (R @ c).T @ g)


def lyap(R, ω, v, Δp):
    return energy(R, ω) + 0.5 * m * v @ v + 0.5 * m * Δp @ Δp


# initial state
r = np.zeros(3)
v = np.zeros(3)
ω = np.zeros(3)
R = np.eye(3)

# R = Rotation.from_rotvec([0.4*np.pi, 0, 0]).as_matrix()
# TODO we cannot have any angular velocity about z
ω = np.array([0.5, 0.3, 1])
# v = np.cross(ω, c)

# print(np.cross(c, I @ c))

# A = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]])
# D = A.T @ A
#
# B = -rg.skew3(c) @ R.T
# K = R @ D @ rg.skew3(c) @ D * 10
#
# vhat = null_space(B @ K).squeeze()
#
# assert np.all(np.linalg.eigvals(B @ K) >= 0)
# # print(B @ K @ c)  # TODO want this to be zero
# print(c.T @ I @ vhat)
# print(c.T @ np.cross(I @ vhat, vhat))

# IPython.embed()
# raise ValueError()

rd = r + [2, -1, 0.1]
pd = rd - c

ts = []
rs = []
vs = []
ωs = []
φs = []
Δps = []
Vs = []
Es = []

ps = []
pdots = []
pddots = []

t = 0
i = 0
while t < DURATION:
    p = r - R @ c
    pdot = R @ (v - np.cross(ω, c))
    Δp = pd - p
    Δr = rd - r

    u = -pdot + Δp
    # u = -v + Δr
    # u = np.zeros(3)
    # u = -(v @ c) * c
    # u = -pdot

    τ = m * np.cross(c, R.T @ (g - u))
    ωdot = np.linalg.solve(I, τ - np.cross(ω, I @ ω) - α * np.outer(ρ, ρ) @ ω)
    # τ = m * np.cross(c, R.T @ g)
    # ωdot = np.linalg.solve(I + LENGTH**2 * np.eye(3), τ - np.cross(ω, I @ ω) - α * np.outer(ρ, ρ) @ ω)

    pddot = u + R @ (np.cross(ωdot, c) + np.cross(ω, np.cross(ω, c)))

    # position
    vdot = R.T @ u - np.cross(ω, v)
    v = v + vdot * TIMESTEP

    rdot = R @ v
    r = r + rdot * TIMESTEP

    # orientation
    ω = ω + ωdot * TIMESTEP
    R = R @ expm(rg.skew3(ω) * TIMESTEP)


    # print(v - np.cross(ω, c))
    # print(np.linalg.norm(v @ np.cross(ω, c)))

    ts.append(t)
    rs.append(r)
    vs.append(v)
    ωs.append(ω)
    φs.append(Rotation.from_matrix(R).as_rotvec())
    Δps.append(Δp)
    Vs.append(lyap(R, ω, v, Δp))
    Es.append(energy(R, ω))

    ps.append(p)
    pdots.append(pdot)
    pddots.append(pddot)

    i += 1
    t = i * TIMESTEP

ts = np.array(ts)
rs = np.array(rs)
vs = np.array(vs)
ωs = np.array(ωs)
φs = np.array(φs)
Δps = np.array(Δps)
ps = np.array(ps)
pdots = np.array(pdots)
pddots = np.array(pddots)

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
plt.title("Position error")
plt.plot(ts, Δps[:, 0], label="x")
plt.plot(ts, Δps[:, 1], label="y")
plt.plot(ts, Δps[:, 2], label="z")
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Δp [m]")
plt.grid()

plt.figure()
plt.title("p")
plt.plot(ts, ps[:, 0], label="x")
plt.plot(ts, ps[:, 1], label="y")
plt.plot(ts, ps[:, 2], label="z")
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("p [m]")
plt.grid()

plt.figure()
plt.title("pdot")
plt.plot(ts, pdots[:, 0], label="x")
plt.plot(ts, pdots[:, 1], label="y")
plt.plot(ts, pdots[:, 2], label="z")
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("pdot [m/s]")
plt.grid()

plt.figure()
plt.title("pddot")
plt.plot(ts, pddots[:, 0], label="x")
plt.plot(ts, pddots[:, 1], label="y")
plt.plot(ts, pddots[:, 2], label="z")
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("pddot [m/s^2]")
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

plt.figure()
plt.title("Energy")
plt.plot(ts, Vs, label="V")
plt.plot(ts, Es, label="E")
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Energy")
plt.grid()

plt.show()
