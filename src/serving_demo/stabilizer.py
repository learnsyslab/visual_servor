import numpy as np
from qpsolvers import solve_qp

import mobile_manipulation_central as mm

import IPython


class PendulumStabilizer:
    def __init__(
        self,
        gain,
        model,
        tray_vel_filter_tau=0.01,
        accel_max=0.5,
        vel_max=0.1,
        joint_vel_max=0.1,
    ):
        self.gain = gain
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

    def reset(self):
        self.v_ee = np.zeros(3)
        self.tray_pos_prev = None
        self.tray_vel_filter.reset()

    def update(self, q, tray_position, dt, solver="quadprog"):
        # estimate tray velocity
        if self.tray_pos_prev is None:
            v_tray_raw = np.zeros(3)
        else:
            v_tray_raw = (tray_position - self.tray_pos_prev) / dt
        self.tray_pos_prev = tray_position
        v_tray = self.tray_vel_filter.update(v_tray_raw, dt)

        # compute acceleration input
        u = self.gain * (v_tray - 2 * self.v_ee)
        u = np.clip(u, -self.accel_max, self.accel_max)

        # integrate to get commanded velocity
        # this is in the world frame
        self.v_ee += dt * u
        self.v_ee = np.clip(self.v_ee, -self.vel_max, self.vel_max)

        # diff IK QP
        J = self.model.jacobian(q)
        ξ_ee = np.concatenate((self.v_ee, np.zeros(3)))
        A = np.hstack((J[:, 3:], -ξ_ee.reshape((6, 1))))

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
