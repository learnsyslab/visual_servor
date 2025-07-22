import numpy as np


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
