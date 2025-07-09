import cvxpy as cp
import numpy as np
import rigeo as rg
from scipy.linalg import expm
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

import serving_demo as sd

import IPython

np.set_printoptions(precision=4, suppress=True)


DURATION = 12
STEPS = DURATION * 100
TIMESTEP = DURATION / STEPS

INTERVAL = DURATION / 3

GRAVITY = np.array([0, 0, -9.81])

TRAY_RADIUS = 0.2
CYLINDER_HEIGHT = 0.001

# height of the object's CoM above its base
OBJ_BASE_HALF_EXTENT = 0.05

# object-tray friction coefficient
MU = 0.5


def simulate(
    pendulum_length=0.3,
    obj_height=0.1,
    accel=3,
    static=False,
    pump_energy=False,
    plot=True,
):
    def input_accel(t):
        if t <= INTERVAL:
            return np.array([accel, 0, 0])
        elif t <= 2 * INTERVAL:
            return np.array([-accel, 0, 0])
        return np.zeros(3)

    tray = rg.Cylinder(
        length=CYLINDER_HEIGHT, radius=TRAY_RADIUS, center=[0, 0, -pendulum_length]
    )
    tray_params = tray.uniform_density_params(mass=0.5)

    half_extents = [OBJ_BASE_HALF_EXTENT, OBJ_BASE_HALF_EXTENT, obj_height / 2]
    box_center = tray.center + [0, 0, 0.5 * (obj_height + CYLINDER_HEIGHT)]
    box = rg.Box(half_extents, center=box_center)
    box_params = box.uniform_density_params(mass=0.5)

    combined_params = tray_params + box_params

    # spatial mass matrices
    M_o = box_params.M
    M_p = tray_params.M

    # contact points
    contacts = np.array(
        [
            [OBJ_BASE_HALF_EXTENT, OBJ_BASE_HALF_EXTENT, -pendulum_length],
            [OBJ_BASE_HALF_EXTENT, -OBJ_BASE_HALF_EXTENT, -pendulum_length],
            [-OBJ_BASE_HALF_EXTENT, -OBJ_BASE_HALF_EXTENT, -pendulum_length],
            [-OBJ_BASE_HALF_EXTENT, OBJ_BASE_HALF_EXTENT, -pendulum_length],
        ]
    )
    Gs = [rg.contact_jacobian(c) for c in contacts]

    mG_o = box_params.mass * rg.contact_jacobian(box_params.com)
    mG_p = tray_params.mass * rg.contact_jacobian(tray_params.com)

    # anchor points for tray ropes
    anchors = np.array(
        [
            [TRAY_RADIUS, 0, -pendulum_length],
            [-TRAY_RADIUS, 0, -pendulum_length],
            [0, -TRAY_RADIUS, -pendulum_length],
            [0, TRAY_RADIUS, -pendulum_length],
        ]
    )

    # friction matrix: F @ f <= 0 means f inside friction cone
    F = np.array([[0, 0, -1], [1, 1, -MU], [1, -1, -MU], [-1, -1, -MU], [-1, 1, -MU]])

    # initial state
    ξ = rg.SV.zero()
    C = np.eye(3)  # this is C_wb

    # spatial acceleration: this can be computed analytically
    ωdot = cp.Parameter(3)
    vdot = cp.Parameter(3)
    ξdot = cp.hstack([ωdot, vdot])

    # remainder wrenches
    rem_p = cp.Parameter(6)
    rem_o = cp.Parameter(6)

    # contact forces
    fc = cp.Variable((4, 3))

    # total contact wrench on the object
    # this is applied the opposite way on the tray
    wc = cp.sum([G @ f for G, f in zip(Gs, fc)])

    # tensile force along ropes
    wt = cp.Variable(6)

    # minimize the shear contact forces
    # W = np.diag([1, 1, 0])
    # objective = cp.Minimize(cp.sum([cp.quad_form(f, W, assume_PSD=True) for f in fc]))
    s = cp.Variable(1)
    objective = cp.Minimize(s)
    # objective = cp.Minimize(cp.sum([cp.norm(f[:2]) for f in fc]))

    # Newton-Euler equations (force-torque balance)
    constraints = [rem_o + wc == M_o @ ξdot, rem_p - wc + wt == M_p @ ξdot]

    # friction constraints
    # constraints.extend([F @ f <= s for f in fc])
    constraints.extend([cp.norm(f[:2]) <= MU * f[2] + s for f in fc])
    # constraints.extend([f[2] >= 0 for f in fc])

    if not static:
        # tensile forces must be positive
        fts = cp.Variable(4, nonneg=True)
        ft = cp.sum([-f * sd.unit(a) for f, a in zip(fts, anchors)])

        # no torque because forces all act through the origin
        constraints.append(wt == cp.hstack((np.zeros(3), ft)))

    problem = cp.Problem(objective, constraints)

    ts = []
    us = []
    φs = []
    ξs = []
    fcs = []
    ss = []

    t_fail = None

    t = 0
    for i in range(STEPS):
        t = i * TIMESTEP
        u = input_accel(t)

        # pump energy into the system
        if pump_energy and t > INTERVAL:
            v_p = ξ.linear + np.cross(ξ.angular, tray_params.com)
            u = -sd.unit(C @ v_p) * accel

        # solve for spatial acceleration
        if static:
            ωdot.value = np.zeros(3)
            vdot.value = C.T @ u
        else:
            ωdot.value = np.linalg.lstsq(
                combined_params.I,
                np.cross(combined_params.h, C.T @ (GRAVITY - u))
                - np.cross(ξ.angular, combined_params.I @ ξ.angular),
                rcond=None,
            )[0]
            vdot.value = C.T @ u - np.cross(ξ.angular, ξ.linear)

        # gravity and Coriolis forces on each body
        rem_p.value = ξ.adjoint().T @ M_p @ ξ.vec + mG_p @ C.T @ GRAVITY
        rem_o.value = ξ.adjoint().T @ M_o @ ξ.vec + mG_o @ C.T @ GRAVITY

        # solve for contact forces
        problem.solve(solver=cp.CLARABEL)
        if problem.status != "optimal":
            print(f"failed to solve at time {t}")
            IPython.embed()
            break

        ss.append(s.value)
        if s.value > 0 and t_fail is None:
            t_fail = t

        # integrate forward in time
        ξ = ξ + rg.SV(linear=vdot.value, angular=ωdot.value) * TIMESTEP
        C = C @ expm(rg.skew3(ξ.angular) * TIMESTEP)

        ts.append(t)
        us.append(u)
        φs.append(Rotation.from_matrix(C).as_rotvec())

        # rotate into the world frame
        ξs.append(np.concatenate((C @ ξ.angular, C @ ξ.linear)))

        fcs.append(fc.value)

    ss = np.array(ss)
    us = np.array(us)
    φs = np.array(φs)
    ξs = np.array(ξs)
    fcs = np.array(fcs)

    # fts = np.linalg.norm(fcs[:, :, :2], axis=2)
    # fns = fcs[:, :, 2]
    # μs = fts / fns
    # μ_max = np.max(μs)
    # print(f"max mu required = {μ_max}")
    # print(f"max s required = {np.max(ss)}")

    # φs[φs < 0] += 2 * np.pi

    if plot:
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
        plt.plot(ts, fcs[:, 0, 0], label="x")
        plt.plot(ts, fcs[:, 0, 1], label="y")
        plt.plot(ts, fcs[:, 0, 2], label="z")
        plt.grid()
        plt.xlabel("Time [s]")
        plt.ylabel("Contact force [N]")
        plt.legend()

        plt.show()

    # max friction constraint violation
    return np.max(ss), t_fail


def main():
    # scale up acceleration
    # TODO should I also test the time intervals?
    # for a in [2, 2.5, 3, 3.5, 4, 4.5, 5]:
    #     s_max, _ = simulate(accel=a, plot=False)
    #     print(f"a = {a}, max s = {s_max}")

    # object height
    # TODO: with a=2 the static approach is better
    # for h in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
    #     s_max, _ = simulate(accel=2, obj_height=h, plot=False)
    #     print(f"h = {h}, max s = {s_max}")

    # for L in [0.2, 0.3, 0.4, 0.5]:
    #     s_max, _ = simulate(pendulum_length=L, plot=False)
    #     print(f"L = {L}, max s = {s_max}")

    s_max, t_fail = simulate(pump_energy=True, plot=False)
    print(f"s = {s_max}, t = {t_fail}")


if __name__ == "__main__":
    main()
