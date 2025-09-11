#!/usr/bin/env python3
from pathlib import Path

import cvxpy as cp
import numpy as np
import rigeo as rg
from scipy.linalg import expm
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

import visual_servor as vs

import IPython

# timing
STEPS_PER_SEC = 100
TIMESTEP = 1 / STEPS_PER_SEC

INTERVAL = 2
STABILIZE_TIME = 8
WAIT_TIME = 2
STABILIZE_START_TIME = 3 * INTERVAL + WAIT_TIME

GRAVITY = np.array([0, 0, -9.81])

# tray is modelled as a cylinder
TRAY_RADIUS = 0.2
TRAY_HEIGHT = 0.001

# object is a box
OBJ_BASE_HALF_EXTENT = 0.05

# object-tray friction coefficient
MU = 0.1

USE_FEEDBACK_CONTROLLER = False


def simulate(
    pendulum_length=0.3,
    obj_height=0.1,
    accel=1,
    max_lqr_accel=1,
    obj_xy_offset=None,
    static=False,
    pump_energy=False,
    stabilize=False,
    point_mass=False,
    use_integral_term=True,
):
    """Simulation the pendulum-object system.

    Parameters
    ----------
    pendulum_length : float
        Length of the pendulum (distance from pivot to top of tray).
    obj_height : float
        Height of the object (which is a box).
    accel : float
        Maximum acceleration of the tray.
    obj_xy_offset : array-like, shape (2,) or None
        Offset of the object from the center of the tray.
    static : bool
        If ``True``, simulate a static tray (no pendulum dynamics).
    pump_energy : bool
        If ``True``, after the first interval, apply acceleration in the
        direction opposite the velocity of the tray.
    stabilize : bool
        If ``True``, after the main trajectory, use LQR to stabilize the
        pendulum.
    point_mass : bool
        If ``True``, model the tray and object as point masses at the
        end of the pendulum.
    use_integral_term : bool
        If ``True``, include an the integral of position error in the LQR
        controller.
    """
    traj = vs.TrapezoidalTrajectory(a=accel, t1=INTERVAL, t2=2 * INTERVAL)

    duration = STABILIZE_START_TIME + STABILIZE_TIME
    # if stabilize:
    #     duration += STABILIZE_TIME

    N = duration * STEPS_PER_SEC

    if obj_xy_offset is None:
        obj_xy_offset = np.zeros(2)
    obj_offset = np.append(obj_xy_offset, 0)

    # NOTE: we define pendulum length as the distance to the top of the tray
    tray_z = -pendulum_length - 0.5 * TRAY_HEIGHT
    tray = rg.Cylinder(length=TRAY_HEIGHT, radius=TRAY_RADIUS, center=[0, 0, tray_z])
    tray_params = tray.uniform_density_params(mass=0.5)

    half_extents = [OBJ_BASE_HALF_EXTENT, OBJ_BASE_HALF_EXTENT, obj_height / 2]
    box_z = -pendulum_length + 0.5 * obj_height
    box = rg.Box(half_extents, center=[0, 0, box_z] + obj_offset)
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
    r = np.zeros(3)
    C = np.eye(3)  # this is C_wb
    rd = r.copy()

    # vw = np.zeros(3)
    r_dot = np.zeros(3)
    ω = np.zeros(3)

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
    # TODO would this actually be fine if the correct offset was used? I think
    # we can explain this away in the paper
    lqr_gain = vs.pendulum_lqr_gain(
        length=pendulum_length, use_integral_term=use_integral_term
    )
    Δr_int = np.zeros(3) if use_integral_term else None

    ts = []
    rs = []
    rds = []
    r_dots = []
    r_trays = []
    r_tray_dots = []
    ωs = []
    us = []
    φs = []
    fcs = []
    ss = []

    Kp = 1
    Kv = 2 * np.sqrt(Kp)

    t = 0
    for i in range(N):
        t = i * TIMESTEP

        v = C.T @ r_dot
        r_tray = r + C @ tray_params.com
        r_tray_dot = r_dot + C @ np.cross(ω, tray_params.com)

        # desired values
        rxd, vxd, axd = traj.sample(t)
        rd = np.array([rxd, 0, 0])
        r_dot_d = np.array([vxd, 0, 0])  # TODO naming is not great
        ad = np.array([axd, 0, 0])

        # basic feedback control
        if USE_FEEDBACK_CONTROLLER:
            u = ad + Kp * (rd - r) + Kv * (r_dot_d - r_dot)
        else:
            u = ad

        # u = np.array([ux, 0, 0])

        # pump energy into the system
        if pump_energy and t > INTERVAL:
            u = -vs.unit(r_tray_dot) * accel

        if stabilize and t > STABILIZE_START_TIME:

            # TODO also add u check here
            # if np.linalg.norm(ξ.vec) < 0.01:
            #     print(f"done stabilizing at t = {t}")
            #     break

            # all quantities in global frame
            Δr = r - rd
            if use_integral_term:
                # TODO could include an anti-windup term here
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
            if u_norm > max_lqr_accel:
                u = u / u_norm * max_lqr_accel

        # solve for spatial acceleration
        if static:
            ωdot.value = np.zeros(3)
            vdot.value = C.T @ u
        else:
            ωdot.value = np.linalg.lstsq(
                combined_params.I,
                np.cross(combined_params.h, C.T @ (GRAVITY - u))
                - np.cross(ω, combined_params.I @ ω),
                rcond=None,
            )[0]
            vdot.value = C.T @ u - np.cross(ω, v)

        # gravity and Coriolis forces on each body
        ξ = rg.SV(linear=v, angular=ω)
        rem_p.value = ξ.adjoint().T @ M_p @ ξ.vec + mG_p @ C.T @ GRAVITY
        rem_o.value = ξ.adjoint().T @ M_o @ ξ.vec + mG_o @ C.T @ GRAVITY

        # solve for contact forces
        problem.solve(solver=cp.CLARABEL)
        if problem.status != "optimal":
            print(f"failed to solve at time {t}")
            IPython.embed()
            break

        # integrate linear part
        r_dot = r_dot + u * TIMESTEP
        r = r + r_dot * TIMESTEP

        # integrate angular part
        ω = ω + ωdot.value * TIMESTEP
        C = C @ expm(rg.skew3(ω) * TIMESTEP)

        ss.append(s.value)
        ts.append(t)
        us.append(u)
        φs.append(Rotation.from_matrix(C).as_rotvec())
        rs.append(r)
        rds.append(rd)
        ωs.append(C @ ω)  # rotate into world frame
        r_dots.append(r_dot)
        fcs.append(fc.value)
        r_trays.append(r_tray)
        r_tray_dots.append(r_tray_dot)

        # break early in the pump energy case, because it wil fail to solve
        # later (and we don't care about the rest of the trajectory)
        if s.value < 0 and pump_energy:
            break

    return vs.SimulationData(
        t_interval=INTERVAL,
        t_wait=WAIT_TIME,
        t_stabilize=STABILIZE_TIME,
        ts=ts,
        ss=ss,
        us=us,
        φs=φs,
        fcs=fcs,
        rs=rs,
        rds=rds,
        r_dots=r_dots,
        ωs=ωs,
        r_trays=r_trays,
        r_tray_dots=r_tray_dots,
    )


def simulate_parameter_error():
    path = Path("data/param_error")
    path.mkdir(parents=True, exist_ok=True)

    # first just compare stabilize vs non-stabilize
    simulate(stabilize=False).save(path / "sim_nostab.npz")
    simulate(stabilize=True, use_integral_term=True).save(path / "sim_stab.npz")

    # now compare parameter error cases (with stabilization)
    error1 = [0.01, 0]
    error2 = [0.05, 0]
    simulate(stabilize=True, use_integral_term=False).save(path / "sim0_noint.npz")
    simulate(stabilize=True, use_integral_term=True).save(path / "sim0_int.npz")

    simulate(stabilize=True, obj_xy_offset=error1, use_integral_term=False).save(
        path / "sim1_noint.npz"
    )
    simulate(stabilize=True, obj_xy_offset=error1, use_integral_term=True).save(
        path / "sim1_int.npz"
    )

    simulate(stabilize=True, obj_xy_offset=error2, use_integral_term=False).save(
        path / "sim2_noint.npz"
    )
    simulate(stabilize=True, obj_xy_offset=error2, use_integral_term=True).save(
        path / "sim2_int.npz"
    )


def simulate_parameter_variation():
    path = Path("data/param_variation")
    path.mkdir(parents=True, exist_ok=True)

    # scale up acceleration
    print("Accelerations...")
    accelerations = [0.5, 1, 1.5, 2, 2.5, 3]
    for a in accelerations:
        simulate(accel=a, stabilize=True, point_mass=True).save(
            path / f"sim_pendulum_pm_a{a}.npz"
        )
        simulate(accel=a, stabilize=True).save(path / f"sim_pendulum_stab_a{a}.npz")
        simulate(accel=a, stabilize=False).save(path / f"sim_pendulum_nostab_a{a}.npz")
        simulate(accel=a, static=True).save(path / f"sim_static_a{a}.npz")

    # object height
    print("Heights...")
    heights = [0.1, 0.15, 0.2, 0.25, 0.3]
    for h in heights:
        simulate(obj_height=h, stabilize=True, point_mass=True).save(
            path / f"sim_pendulum_pm_h{h}.npz"
        )
        simulate(obj_height=h, stabilize=True).save(
            path / f"sim_pendulum_stab_h{h}.npz"
        )
        simulate(obj_height=h, stabilize=False).save(
            path / f"sim_pendulum_nostab_h{h}.npz"
        )
        simulate(obj_height=h, static=True).save(path / f"sim_static_h{h}.npz")

    # pendulum lengths
    print("Lengths...")
    for L in [0.2, 0.3, 0.4, 0.5, 0.6]:
        simulate(pendulum_length=L, stabilize=True, point_mass=True).save(
            path / f"sim_pendulum_pm_L{L}.npz"
        )
        simulate(pendulum_length=L, stabilize=True).save(
            path / f"sim_pendulum_stab_L{L}.npz"
        )
        simulate(pendulum_length=L, stabilize=False).save(
            path / f"sim_pendulum_nostab_L{L}.npz"
        )


def simulate_pump_energy():
    data = simulate(pump_energy=True)
    print(f"Pump energy failed at t = {data.t_fail} sec")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    # print(f"max static acceleration = {MU * np.linalg.norm(GRAVITY)}")
    # s_min, t_fail = simulate(stabilize=True, plot=True)
    # print(f"s = {s_min}, t = {t_fail}")

    # simulate_parameter_error()
    simulate_parameter_variation()
    # simulate_pump_energy()
