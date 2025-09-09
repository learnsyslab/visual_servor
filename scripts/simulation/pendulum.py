import cvxpy as cp
import numpy as np
import rigeo as rg
from scipy.linalg import expm
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

import visual_servor as vs

import IPython

np.set_printoptions(precision=4, suppress=True)


STEPS_PER_SEC = 100
TIMESTEP = 1 / STEPS_PER_SEC

INTERVAL = 2
STABILIZE_TIME = 20
WAIT_TIME = 2
BASE_DURATION = 3 * INTERVAL + WAIT_TIME

GRAVITY = np.array([0, 0, -9.81])

# tray is modelled as a cylinder
TRAY_RADIUS = 0.2
TRAY_HEIGHT = 0.001
OBJ_BASE_HALF_EXTENT = 0.05

# object-tray friction coefficient
MU = 0.25


def simulate(
    pendulum_length=0.3,
    obj_height=0.1,
    accel=3,
    static=False,
    pump_energy=False,
    stabilize=False,
    point_mass=False,
    use_integral_term=True,
    plot=False,
):
    """TODO"""

    def input_accel(t):
        """World-frame tray acceleration.

        Constant acceleration, zero acceleration, constant negative acceleration.
        """
        if t <= INTERVAL:
            return np.array([accel, 0, 0])
        elif t <= 2 * INTERVAL:
            return np.zeros(3)
        elif t <= 3 * INTERVAL:
            return np.array([-accel, 0, 0])
        return np.zeros(3)

    duration = BASE_DURATION
    if stabilize:
        duration += STABILIZE_TIME

    N = duration * STEPS_PER_SEC

    # NOTE: we define pendulum length as the distance to the top of the tray

    tray_z = -pendulum_length - 0.5 * TRAY_HEIGHT
    tray = rg.Cylinder(length=TRAY_HEIGHT, radius=TRAY_RADIUS, center=[0, 0, tray_z])
    tray_params = tray.uniform_density_params(mass=0.5)

    half_extents = [OBJ_BASE_HALF_EXTENT, OBJ_BASE_HALF_EXTENT, obj_height / 2]
    box_z = -pendulum_length + 0.5 * obj_height
    box = rg.Box(half_extents, center=[0, 0, box_z])
    box_params = box.uniform_density_params(mass=0.5)

    if point_mass:
        point_mass_position = np.array([0, 0, -pendulum_length])
        tray_params = rg.InertialParameters(
            mass=0.5,
            com=point_mass_position,
            H=np.zeros((3, 3)),
            translate_from_com=True,
        )
        box_params = rg.InertialParameters(
            mass=0.5,
            com=point_mass_position,
            H=np.zeros((3, 3)),
            translate_from_com=True,
        )

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
    # F = np.array([[0, 0, -1], [1, 1, -MU], [1, -1, -MU], [-1, -1, -MU], [-1, 1, -MU]])

    # initial state
    ξ = rg.SV.zero()
    r = np.zeros(3)
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
    s = cp.Variable(1)
    objective = cp.Maximize(s)

    # Newton-Euler equations (force-torque balance)
    constraints = [rem_o + wc == M_o @ ξdot, rem_p - wc + wt == M_p @ ξdot]

    # friction constraints
    # constraints.extend([F @ f <= s for f in fc])
    constraints.extend([MU * f[2] - cp.norm(f[:2]) >= s for f in fc])
    # constraints.extend([f[2] >= 0 for f in fc])

    if not static:
        # tensile forces must be positive
        fts = cp.Variable(4, nonneg=True)
        ft = cp.sum([-f * vs.unit(a) for f, a in zip(fts, anchors)])

        # no torque because forces all act through the origin
        constraints.append(wt == cp.hstack((np.zeros(3), ft)))

    problem = cp.Problem(objective, constraints)

    # LQR
    lqr_gain = vs.pendulum_lqr_gain(
        length=pendulum_length, use_integral_term=use_integral_term
    )
    stabilizing = False
    Δr_int = np.zeros(3) if use_integral_term else None

    ts = []
    us = []
    φs = []
    ξs = []
    fcs = []
    ss = []

    t_fail = None

    t = 0
    for i in range(N):
        t = i * TIMESTEP
        u = input_accel(t)

        r_tray = r + C @ tray_params.com
        r_tray_dot = C @ (ξ.linear + np.cross(ξ.angular, tray_params.com))

        # pump energy into the system
        if pump_energy and t > INTERVAL:
            u = -vs.unit(r_tray_dot) * accel

        if stabilize and t > BASE_DURATION:
            # initialize stabilization
            if not stabilizing:
                r_d = r.copy()
            stabilizing = True

            # TODO also add u check here
            if np.linalg.norm(ξ.vec) < 0.01:
                print(f"done stabilizing at t = {t}")
                break

            r_dot = C @ ξ.linear

            # all quantities in global frame
            Δr = r - r_d
            if use_integral_term:
                Δr_int = Δr_int + TIMESTEP * Δr
            ρ = (r_tray - r) / pendulum_length
            ρ_dot = (r_tray_dot - r_dot) / pendulum_length

            x = vs.pendulum_lqr_state(
                Δr=Δr, ρ=ρ, v_ee=r_dot, ρ_dot=ρ_dot, Δr_int=Δr_int
            )

            u = np.zeros(3)
            u[:2] = -lqr_gain @ x

            # limit acceleration
            u_norm = np.linalg.norm(u)
            if u_norm > accel:
                u = u / u_norm * accel

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
        if s.value < 0 and t_fail is None:
            t_fail = t
            # TODO may need to break in the pump_energy case
            # break

        # integrate forward in time
        ξ = ξ + rg.SV(linear=vdot.value, angular=ωdot.value) * TIMESTEP
        r = r + C @ ξ.linear * TIMESTEP
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

    fts = np.linalg.norm(fcs[:, :, :2], axis=2)
    fns = fcs[:, :, 2]
    μs = fts / fns
    μ_max = np.max(μs)
    s_min = np.min(ss)

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
    return s_min, t_fail, μ_max


def main():
    # s_min, t_fail = simulate(stabilize=True, plot=True)
    # print(f"s = {s_min}, t = {t_fail}")

    print(f"max static acceleration = {MU * np.linalg.norm(GRAVITY)}")

    # TODO should I also test the time intervals - that is, the trajectory
    # shape? this is probably too complicated

    # TODO also compare stabilize vs non-stabilize

    # scale up acceleration
    # accelerations = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    # print("\nHanging")
    # for a in accelerations:
    #     s_min = simulate(accel=a, stabilize=True)[0]
    #     print(f"a = {a}, s_min = {s_min}")
    # print("\nStatic")
    # for a in accelerations:
    #     s_min = simulate(accel=a, static=True)[0]
    #     print(f"a = {a}, s_min = {s_min}")

    # object height
    # print("\nHanging")
    # heights = [0.1, 0.15, 0.2, 0.25, 0.3]
    # for h in heights:
    #     s_min = simulate(accel=2, obj_height=h, stabilize=True)[0]
    #     print(f"h = {h}, s_min = {s_min}")
    # print("\nStatic")
    # for h in heights:
    #     s_min = simulate(accel=2, obj_height=h, static=True)[0]
    #     print(f"h = {h}, s_min = {s_min}")

    # pendulum lengths
    print("\nHanging")
    for L in [0.2, 0.3, 0.4, 0.5, 0.6]:
        s_min = simulate(pendulum_length=L, stabilize=True)[0]
        print(f"L = {L}, s_min = {s_min}")

    # failure case
    # s_min, t_fail, _ = simulate(pump_energy=True)
    # print(f"s = {s_min}, t = {t_fail}")


if __name__ == "__main__":
    main()
