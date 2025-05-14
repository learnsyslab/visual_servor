import numpy as np


def unit(x):
    norm = np.linalg.norm(x)
    if norm > 0:
        return x / norm
    return x


def orth(v):
    """Generate a 2D orthogonal to v."""
    return np.array([v[1], -v[0]])
