import numpy as np
from qpsolvers import solve_qp
from lpsolvers import solve_lp


class CollisionEllipse:
    def __init__(self, rx=0.5, ry=0.5, center=None):
        if center is None:
            center = np.zeros(2)

        self.ell_shape = np.diag([1.0 / rx**2, 1.0 / ry**2])
        self.ell_center = center

    def squared_dist(self, points):
        x = points - self.ell_center
        y = x @ self.ell_shape
        return np.sum(x * y, axis=1)

    def process_scan(self, scan, lidar_offset=None, num_buckets=20):
        if lidar_offset is None:
            lidar_offset = np.zeros(2)

        n = len(scan.ranges)
        num_per_bucket = n // num_buckets

        ranges = np.array(scan.ranges)
        angles = np.array([scan.angle_min + i * scan.angle_increment for i in range(n)])
        points = (np.vstack((np.cos(angles), np.sin(angles))) * ranges).T

        # wrt the center of the ellipse
        points = points + lidar_offset

        # compute squared Mahalanobis distances
        dists = self.squared_dist(points)

        # remove invalid points
        valid = (ranges >= scan.range_min) & (ranges <= scan.range_max)
        dists[~valid] = np.inf

        bucketed_points = []
        for i in range(num_buckets):
            s = i * num_per_bucket
            e = min((i + 1) * num_per_bucket, n)
            min_idx = np.argmin(dists[s:e]) + s
            min_dist = dists[min_idx]

            # only use the point if it is within the ellipse; otherwise there
            # are no relevant points in this bucket
            if min_dist <= 1:
                bucketed_points.append(points[min_idx, :])

        return np.array(bucketed_points)

    def filter_safe_velocity(self, base_vel_des, points, lb, ub, solver="quadprog"):
        """Compute a velocity as close as possible to this one that avoids
        collisions.

        Parameters
        ----------
        base_vel_des : np.ndarray, shape (3,)
            The desired planar velocity twist of the base.
        points : np.ndarray, shape (n, 2)
            The points with which to avoid collisions. It is assumed that these
            have already been processed to remove points outside the ellipsoid.
        """
        n = len(points)
        if n == 0:
            return base_vel_des

        # compute the normal direction to each point (i.e., the direction of
        # the shortest distance to the ellipse boundary)
        y = (points - self.ell_center) @ self.ell_shape
        normals = y / np.linalg.norm(y, axis=1)[:, None]

        P = np.eye(3)
        h = np.zeros(n)

        S = np.array([[0, -1], [1, 0]])
        zs = np.sum(normals * (S @ points.T).T, axis=1)
        G = np.hstack((normals, zs[:, None]))

        # no need to solve the QP if no constraints active
        if np.all(G @ base_vel_des <= h):
            return base_vel_des

        # TODO do I need to use a dedicated class to improve speed?
        # x = solve_qp(P=P, q=-base_vel_des, G=G, h=h, solver=solver)

        # NOTE: did some experimenting with an LP formulation, but so far we
        # aren't using collision avoidance at all
        # G = np.vstack((G, np.eye(3), -np.eye(3)))
        # if base_vel_des[2] >= 0:
        #     h = np.concatenate((h, [ub[0], ub[1], base_vel_des[2]], [-lb[0], -lb[1], 0]))
        # else:
        #     h = np.concatenate((h, [ub[0], ub[1], 0], [-lb[0], -lb[1], -base_vel_des[2]]))
        # x = solve_lp(c=-base_vel_des, G=G, h=h, solver="cvxopt")

        if x is None:
            print("failed to solve obstacle avoidance QP")
            return np.zeros_like(base_vel_des)
        return x
