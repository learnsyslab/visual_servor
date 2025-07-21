import numpy as np
from qpsolvers import solve_qp


class CollisionEllipse:
    def __init__(self, rx=0.5, ry=0.5, center=None):
        if center is None:
            center = np.zeros(2)

        self.ell_shape = np.diag([1.0 / rx**2, 1.0 / ry**2])
        self.ell_center = center

    def _bucket_points(points, n=20):
        pass

    def squared_dist(points):
        x = points - self.ell_center
        y = x @ self.ell_shape
        return np.sum(x * y, axis=1)

    def filter_safe_velocity(self, lin_vel, ang_vel, points, solver="quadprog"):
        if len(points) == 0:
            return lin_vel, ang_vel

        # remove points outside of the influence ellipse
        x = points - self.ell_center
        y = x @ self.ell_shape
        valid = np.sum(x * y, axis=1) <= 1

        y = y[valid, :]
        points = points[valid, :]

        n = len(points)
        if n == 0:
            # none of the points are inside the ellipse
            return lin_vel, ang_vel

        normals = y / np.linalg.norm(y, axis=1)[:, None]

        P = np.eye(3)
        ξd = np.append(lin_vel, ang_vel)
        h = np.zeros(n)

        # TODO we can bucket the points to avoid so many constraints
        S = np.array([[0, -1], [1, 0]])
        zs = np.sum(normals * (S @ points.T).T, axis=1)
        G = np.hstack((normals, zs[:, None]))

        # no need to solve the QP if no constraints active
        if np.all(G @ ξd <= h):
            return lin_vel, ang_vel

        x = solve_qp(P=P, q=-ξd, G=G, h=h, solver=solver)
        if x is None:
            print("failed to solve obstacle avoidance QP")
            return np.zeros(2), 0
        return x[:2], x[2]
