import cvxpy as cp
import numpy as np
import rigeo as rg
import matplotlib.pyplot as plt
from qpsolvers import solve_qp
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

import serving_demo as sd
import mobile_manipulation_central as mm

import IPython

np.set_printoptions(precision=4, suppress=True)


INTERVAL = 4
STEPS_PER_SEC = 100
TIMESTEP = 1 / STEPS_PER_SEC

GRAVITY = np.array([0, 0, -9.81])

TRAY_RADIUS = 0.2
CYLINDER_HEIGHT = 0.001
OBJ_BASE_HALF_EXTENT = 0.05

# object-tray friction coefficient
MU = 0.5


class DiffIKQP:
    def __init__(self, model, joint_vel_max=0.1):
        self.model = model

        # constant QP data matrices
        self.P = np.diag(np.append(np.ones(6), 0.01))
        self.q = np.append(np.zeros(6), -1)
        self.ub = np.append(joint_vel_max * np.ones(6), 1)
        self.lb = np.append(-joint_vel_max * np.ones(6), 0)
        self.b = np.zeros(6)

    def solve(self, q, v_ee, solver="quadprog"):
        J = self.model.jacobian(q)
        ξ_ee = np.concatenate((v_ee, np.zeros(3)))
        A = np.hstack((J[:, 3:], -ξ_ee.reshape((6, 1))))

        x = solve_qp(
            P=self.P,
            q=self.q,
            A=A,
            b=self.b,
            lb=self.lb,
            ub=self.ub,
            solver=solver,
            verbose=False,
        )
        if x is None:
            return None
        return x[:6]


def simulate(
    pendulum_length=0.3,
    obj_height=0.1,
    accel=3,
    static=False,
    pump_energy=False,
    stabilize=False,
    plot=True,
):

    model = mm.MobileManipulatorKinematics(tool_link_name="ur10_arm_tool0")
    qp = DiffIKQP(model)
    q = np.array([0, 0, 0, 1.5708, -1.5708, 0.7854, 0.7854, 1.5708, -0.2618])

    duration = 5 * INTERVAL if stabilize else 3 * INTERVAL
    N = duration * STEPS_PER_SEC

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

    model.forward(q)
    r = model.link_pose()[0]

    r_d = r.copy()
    v_ee = np.zeros(3)

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
    objective = cp.Minimize(s)

    # Newton-Euler equations (force-torque balance)
    constraints = [rem_o + wc == M_o @ ξdot, rem_p - wc + wt == M_p @ ξdot]

    # friction constraints
    constraints.extend([cp.norm(f[:2]) <= MU * f[2] + s for f in fc])

    if not static:
        # tensile forces must be positive
        fts = cp.Variable(4, nonneg=True)
        ft = cp.sum([-f * sd.unit(a) for f, a in zip(fts, anchors)])

        # no torque because forces all act through the origin
        constraints.append(wt == cp.hstack((np.zeros(3), ft)))

    problem = cp.Problem(objective, constraints)

    # LQR
    lqr_gain = sd.pendulum_lqr_gain(length=pendulum_length, use_integral_term=True)

    ts = []
    us = []
    φs = []
    ξs = []
    fcs = []
    ss = []
    rs = []

    t_fail = None

    Δr_int = np.zeros(3)

    t = 0
    for i in range(N):
        t = i * TIMESTEP

        # CoM error results in significant position error
        com = tray_params.com + [0.01, 0.01, 0]
        r_tray = r + C @ com
        r_tray_dot = C @ (ξ.linear + np.cross(ξ.angular, com))

        # all quantities in global frame
        Δr = r - r_d
        Δr_int = Δr_int + TIMESTEP * Δr
        ρ = (r_tray - r) / pendulum_length
        ρ_dot = (r_tray_dot - v_ee) / pendulum_length

        # LQR state
        x = sd.pendulum_lqr_state(Δr=Δr, ρ=ρ, v_ee=v_ee, ρ_dot=ρ_dot, Δr_int=Δr_int)
        print(f"x = {x}")

        u = np.zeros(3)
        u[:2] = -lqr_gain @ x

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

        # forward integrate
        v_ee = v_ee + u * TIMESTEP
        ω = ξ.angular + ωdot.value * TIMESTEP
        ξ = rg.SV(linear=C.T @ v_ee, angular=ω)

        # diff IK QP
        cmd_vel = qp.solve(q, v_ee)
        if cmd_vel is None:
            print("failed to solve QP")
            IPython.embed()
            break

        q = q + np.concatenate((np.zeros(3), cmd_vel * TIMESTEP))
        model.forward(q)
        r = model.link_pose()[0]

        C = C @ expm(rg.skew3(ξ.angular) * TIMESTEP)

        ts.append(t)
        us.append(u)
        φs.append(Rotation.from_matrix(C).as_rotvec())

        # rotate into the world frame
        ξs.append(np.concatenate((C @ ξ.angular, C @ ξ.linear)))

        fcs.append(fc.value)
        rs.append(r - r_d)

    ss = np.array(ss)
    us = np.array(us)
    φs = np.array(φs)
    ξs = np.array(ξs)
    fcs = np.array(fcs)
    rs = np.array(rs)

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
        plt.title("EE Position")
        plt.plot(ts, rs[:, 0], label="x")
        plt.plot(ts, rs[:, 1], label="y")
        plt.plot(ts, rs[:, 2], label="z")
        plt.grid()
        plt.xlabel("Time [s]")
        plt.ylabel("Position [m]")
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
    s_max, t_fail = simulate(plot=True)
    print(f"s = {s_max}, t = {t_fail}")


if __name__ == "__main__":
    main()
