import numpy as np
from qpsolvers import solve_qp
from scipy.linalg import solve_continuous_are
from spatialmath.base import rotz

import rigeo as rg
import mobile_manipulation_central as mm

import IPython


def pendulum_lqr_gain(length, gravity=-9.81, use_integral_term=False):
    """Compute the LQR gain matrix ``K`` for the 2D pendulum.

    This yields the linear control law ``u = -K @ x``.

    Parameters
    ----------
    length : float, non-negative
        The length of the pendulum.
    gravity : float
        Gravity value.
    use_integral_term : bool
        Set ``True`` to add an extra term to the system which is the integral
        of the EE position. This helps eliminate steady-state error in the
        presence of model error.

    Returns
    -------
    : np.ndarray
        The gain matrix ``K``.
    """
    assert length >= 0
    ρ = np.array([0, 0, -1])
    g = np.array([0, 0, gravity])

    n = 8
    if use_integral_term:
        n += 2

    # x-y components of each of r, ρ, rdot, ρdot
    A = np.zeros((n, n))
    A[0:2, 4:6] = np.eye(2)
    A[2:4, 6:8] = np.eye(2)
    A[6:8, 2:4] = ((rg.skew3(np.cross(ρ, g)) + rg.skew3(ρ) @ rg.skew3(g)) / length)[
        :2, :2
    ]
    if use_integral_term:
        A[8:10, 0:2] = np.eye(2)

    B = np.zeros((n, 2))
    B[4:6, :] = np.eye(2)
    B[6:8, :] = (rg.skew3(ρ) @ rg.skew3(ρ) / length)[:2, :2]

    # solve for feedback gain u = -K @ x with LQR
    Q = np.eye(n)
    R = 0.1 * np.eye(2)
    P = solve_continuous_are(A, B, Q, R)
    return np.linalg.solve(R, B.T @ P)


def pendulum_lqr_state(Δr, ρ, v_ee, ρ_dot, Δr_int=None):
    """Construct the LQR state vector ``x``."""
    x = np.concatenate((Δr[:2], ρ[:2], v_ee[:2], ρ_dot[:2]))
    if Δr_int is not None:
        x = np.concatenate((x, Δr_int[:2]))
    return x


class PendulumStabilizer:
    def __init__(
        self,
        model,
        tray_vel_filter_tau=0.01,
        accel_max=0.5,
        vel_max=0.1,
        joint_vel_max=0.1,
        use_integral_term=False,
    ):
        self.model = model
        self.tray_vel_filter = mm.ExponentialSmoother(τ=tray_vel_filter_tau)
        self.use_integral_term = use_integral_term

        self.accel_max = accel_max
        self.vel_max = vel_max

        # QP data matrices
        self.P = np.diag(np.append(np.ones(6), 0.01))
        self.q = np.append(np.zeros(6), -1)
        self.ub = np.append(joint_vel_max * np.ones(6), 1)
        self.lb = np.append(-joint_vel_max * np.ones(6), 0)
        self.b = np.zeros(6)

        self.tray_pos_prev = None
        self.v_ee = np.zeros(3)
        self.Δr_int = np.zeros(3) if self.use_integral_term else None

    def init(self, q0, r_te_e):
        # compute offset corresponding to tray origin
        self.model.forward(q0)
        self.r_ee_d = self.model.link_pose()[0]

        # offset of actual tray center compared to vicon model tray center
        self.tray_offset = np.append(-r_te_e[:2], 0)
        self.length = np.abs(r_te_e[2]) - 0.1  # TODO

        self.lqr_gain = pendulum_lqr_gain(
            length=self.length, use_integral_term=self.use_integral_term
        )

    def reset(self, q):
        self.model.forward(q)
        self.r_ee_d = self.model.link_pose()[0]
        self.Δr_int = np.zeros(3) if self.use_integral_term else None

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

    def _compute_input_lqr(self, r_ew_w, r_tw_w, v_tray, dt):
        Δr = r_ew_w - self.r_ee_d
        if self.use_integral_term:
            self.Δr_int = self.Δr_int + dt * Δr
            self.Δr_int = np.clip(self.Δr_int, -0.25, 0.25)
        ρ = (r_tw_w - r_ew_w) / self.length
        ρ_dot = (v_tray - self.v_ee) / self.length

        x = pendulum_lqr_state(
            Δr=Δr, ρ=ρ, v_ee=self.v_ee, ρ_dot=ρ_dot, Δr_int=self.Δr_int
        )

        u = np.zeros(3)
        u[:2] = -self.lqr_gain @ x

        print(f"x = {x}")
        print(f"u = {u}")
        # raise ValueError()

        return u

    def update(self, q, tray_position, dt, solver="quadprog"):
        # estimate tray velocity
        v_tray = self._estimate_tray_vel(tray_position, dt)

        self.model.forward(q)
        r_ew_w = self.model.link_pose()[0]
        C_we = rotz(q[2])
        r_tw_w = tray_position + C_we @ self.tray_offset

        # compute acceleration input
        u = self._compute_input_lqr(r_ew_w, r_tw_w, v_tray, dt)
        u = np.clip(u, -self.accel_max, self.accel_max)

        # integrate to get commanded velocity (in the world frame)
        self.v_ee += dt * u
        self.v_ee = np.clip(self.v_ee, -self.vel_max, self.vel_max)

        # diff IK QP
        J = self.model.jacobian(q)
        ξ_ee = np.concatenate((self.v_ee, np.zeros(3)))
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
        # elif (
        #     t > self.min_time
        #     and v is not None
        #     and np.linalg.norm(v) <= self.tray_vel_tol
        # ):
        #     print("tray has converged")
        #     self._active = False
        return self._active
