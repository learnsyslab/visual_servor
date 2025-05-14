"""Test invariance of 3D pendulum control laws using optimization.

Tests reveal that including position error in the control law (at least in the
way I've currently done so), does not work: the origin is not invariant.
"""
import numpy as np
from scipy.linalg import expm, null_space
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import rigeo as rg
import cvxpy as cp

import IPython


MASS = 1
LENGTH = 0.3
INERTIA_COM = np.diag([0.01, 0.02, 0.03])
# INERTIA_COM = 0.1 * np.array([[1, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 1]])
assert np.all(np.linalg.eigvals(INERTIA_COM) >= 0)
GRAVITY = -9.81

# parameters
m = MASS
c = np.array([0, 0, -LENGTH])
I = INERTIA_COM - m * rg.skew3(c) @ rg.skew3(c)
g = np.array([0, 0, GRAVITY])
ρ = c / LENGTH
α = 0.1  # damping about ρ


# initial state
v = np.zeros(3)
ω = np.zeros(3)
R = np.eye(3)

R = Rotation.from_rotvec([0.00*np.pi, 0.001*np.pi, 0]).as_matrix()
ω = np.array([0, 0, 0])
ω[2] = 0  # no angular velocity about z
v = np.cross(ω, c)

S = np.eye(3)
S[2, 2] = 0

ωdot = cp.Variable(3)
Δp = cp.Variable(3)

u = S @ Δp
τ = m * np.cross(c, R.T @ g) - m * rg.skew3(c) @ R.T @ u

d = np.array([1, 0, 0])
objective = cp.Maximize(d @ ωdot)
constraints = [
    I @ ωdot == τ - np.cross(ω, I @ ω),
    c @ ωdot == 0,
    c @ Δp == 0,
    u == S @ R @ (rg.skew3(c) @ ωdot + np.cross(ω, v)),
    # rg.skew3(c) @ ωdot == R.T @ Δp - np.cross(ω, v),
]
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.MOSEK)
print(problem.status)
print(f"ωdot = {ωdot.value}")
print(f"Δp = {Δp.value}")

IPython.embed()
