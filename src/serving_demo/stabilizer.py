import numpy as np
from qpsolvers import solve_qp
from scipy.linalg import solve_continuous_are

import rigeo as rg
import mobile_manipulation_central as mm

import IPython


def pendulum_lqr_gain(length, gravity=-9.81):
    ρ = np.array([0, 0, -1])
    g = np.array([0, 0, gravity])

    # x-y components of each of r, ρ, rdot, ρdot
    A = np.zeros((8, 8))
    A[0:2, 4:6] = np.eye(2)
    A[2:4, 6:8] = np.eye(2)
    A[6:8, 2:4] = ((rg.skew3(np.cross(ρ, g)) + rg.skew3(ρ) @ rg.skew3(g)) / length)[
        :2, :2
    ]

    B = np.zeros((8, 2))
    B[4:6, :] = np.eye(2)
    B[6:8, :] = (rg.skew3(ρ) @ rg.skew3(ρ) / length)[:2, :2]

    # solve for feedback gain u = -K @ x with LQR
    Q = np.eye(8)
    R = 0.1 * np.eye(2)
    P = solve_continuous_are(A, B, Q, R)
    return np.linalg.solve(R, B.T @ P)


class PendulumStabilizer:
    def __init__(
        self,
        model,
        tray_vel_filter_tau=0.01,
        accel_max=0.5,
        vel_max=0.1,
        joint_vel_max=0.1,
    ):
        self.model = model
        self.tray_vel_filter = mm.ExponentialSmoother(τ=tray_vel_filter_tau)

        self.accel_max = accel_max
        self.vel_max = vel_max

        self.P = np.diag(np.append(np.ones(6), 0.01))
        self.q = np.append(np.zeros(6), -1)
        self.ub = np.append(joint_vel_max * np.ones(6), 1)
        self.lb = np.append(-joint_vel_max * np.ones(6), 0)
        self.b = np.zeros(6)

        self.tray_pos_prev = None
        self.v_ee = np.zeros(3)

    def init(self, q0, r_tray_ee):
        # compute offset corresponding to tray origin
        self.model.forward(q0)
        self.r_ee_d, _ = self.model.link_pose()

        self.tray_offset = np.append(-r_tray_ee[:2], 0)
        self.length = np.abs(r_tray_ee[2])

        self.lqr_gain = pendulum_lqr_gain(length=self.length)

    def reset(self, q):
        self.model.forward(q)
        self.r_ee_d, _ = self.model.link_pose()

        self.tray_pos_prev = None
        self.v_ee = np.zeros(3)
        self.tray_vel_filter.reset()

    def _estimate_tray_vel(self, tray_position, dt):
        if self.tray_pos_prev is None:
            v_tray_raw = np.zeros(3)
        else:
            v_tray_raw = (tray_position - self.tray_pos_prev) / dt
        self.tray_pos_prev = tray_position
        return self.tray_vel_filter.update(v_tray_raw, dt)

    def _compute_input_lqr(self, q, r_tray, v_tray):
        self.model.forward(q)
        r_ee, _ = self.model.link_pose()
        Δr = r_ee - self.r_ee_d
        ρ = (r_tray - r_ee) / self.length
        ρdot = (v_tray - self.v_ee) / self.length

        # LQR state
        x = np.concatenate((Δr[:2], ρ[:2], self.v_ee[:2], ρdot[:2]))

        u = np.zeros(3)
        u[:2] = -self.lqr_gain @ x
        return u

    def update(self, q, tray_position, dt, solver="quadprog"):
        # estimate tray velocity
        v_tray = self._estimate_tray_vel(tray_position, dt)

        # compute acceleration input
        r_tray = tray_position + self.tray_offset
        u = self._compute_input_lqr(q, r_tray, v_tray)
        u = np.clip(u, -self.accel_max, self.accel_max)

        # integrate to get commanded velocity
        # this is in the world frame
        self.v_ee += dt * u
        self.v_ee = np.clip(self.v_ee, -self.vel_max, self.vel_max)

        # diff IK QP
        J = self.model.jacobian(q)
        ξ_ee = np.concatenate((self.v_ee, np.zeros(3)))
        A = np.hstack((J[:, 3:], -ξ_ee.reshape((6, 1))))

        # TODO we may also want to steer back toward q0 rather than just desired r
        # TODO use problem class?
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
        arm_cmd_vel = x[:6]
        return np.concatenate((np.zeros(3), arm_cmd_vel))


class PendulumStabilizerTimer:
    def __init__(self, stabilizer, min_time, max_time, tray_vel_tol):
        self.stabilizer = stabilizer
        self.min_time = min_time
        self.max_time = max_time
        self.tray_vel_tol = tray_vel_tol
        self._active = False

    def activate(self):
        self._active = True

    def is_active(self, t):
        """Returns True if the stabilizer is still active, False otherwise."""
        if not self._active:
            return False

        v = self.stabilizer.tray_vel_filter.x

        # max time has elapsed
        if t > self.max_time:
            self._active = False
        elif (
            t > self.min_time
            and v is not None
            and np.linalg.norm(v) <= self.tray_vel_tol
        ):
            print("tray has converged")
            self._active = False
        return self._active
