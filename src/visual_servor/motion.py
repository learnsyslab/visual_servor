import numpy as np


class TrapezoidalTrajectory:
    def __init__(self, a, t1, t2=None):
        if t2 is None:
            t2 = t1
        assert t2 >= t1
        # assumed symmetric: t1 and t2 are the vertices
        self.t1 = t1
        self.t2 = t2
        self.duration = t1 + t2

        # acceleration
        self.a = a

        # peak/constant velocity
        self.v = self.t1 * self.a

        # distances
        self.q1 = 0.5 * self.a * self.t1**2
        self.q2 = self.q1 + (t2 - t1) * self.v
        self.qf = self.q2 + self.q1

    def sample(self, t):
        if t <= self.t1:
            # acceleration
            vd = self.a * t
            qd = 0.5 * self.a * t**2
            ad = self.a
        elif t <= self.t2:
            # constant velocity
            vd = self.v
            qd = self.q1 + (t - self.t1) * self.v
            ad = 0
        elif t <= self.duration:
            # deceleration
            s = t - self.t2
            vd = self.v - s * self.a
            qd = self.q2 + self.v * s - 0.5 * self.a * s**2
            ad = -self.a
        else:
            # end position
            vd = 0
            qd = self.qf
            ad = 0
        return qd, vd, ad


def change_velocity(v, vd, max_a, dt):
    """Accelerate to a desired velocity, with limits.

    Parameters
    ----------
    v : float or np.ndarray
        Current velocity.
    vd : float or np.ndarray, same shape as ``v``
        Desired velocity.
    max_a : float or np.ndarray
        Maximum acceleration. Can either be the same shape as ``v``, or a
        single scalar to use the same value for all dimensions.
    dt : float
        Timestep used to integrate forward in time.

    Returns
    -------
    : float or np.ndarray, same shape as ``v``
        The new velocity.
    """
    # scalar case needs a bit of extra care
    scalar = np.isscalar(v)

    v = np.atleast_1d(v)
    vd = np.atleast_1d(vd)
    assert v.shape == vd.shape

    # accelerate toward desired velocity as fast as possible
    error = vd - v
    new_v = v + dt * np.sign(error) * max_a
    new_error = vd - new_v

    # if any dimension has crossed the reference, then set it to the reference
    crossed_vd = np.sign(error) != np.sign(new_error)
    v = new_v
    v[crossed_vd] = vd[crossed_vd]

    if scalar:
        return v[0]
    return v


def decelerate(v, max_a, dt):
    """Decelerate to zero velocity subject to maximum acceleration."""
    return change_velocity(v=v, vd=np.zeros_like(v), max_a=max_a, dt=dt)
