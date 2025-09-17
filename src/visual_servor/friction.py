from dataclasses import dataclass

import numpy as np
import rosbag
from spatialmath.base import q2r
from mobile_manipulation_central import ros_utils


@dataclass
class FrictionData:
    ts: np.ndarray
    angles: np.ndarray
    mus: np.ndarray
    slips: np.ndarray
    mu: float
    slip_time: float


def quat_to_rot(q):
    """Convert quaternion q to rotation matrix."""
    return q2r(q, order="xyzs")


def compute_friction_data(
    bag_path,
    vicon_tray_name,
    vicon_object_name,
    ignore_first_seconds=0,
    slip_margin=0.005,
):
    bag = rosbag.Bag(bag_path)

    tray_topic = ros_utils.vicon_topic_name(vicon_tray_name)
    tray_msgs = [msg for _, msg, _ in bag.read_messages(tray_topic)]
    ts, tray_poses = ros_utils.parse_transform_stamped_msgs(
        tray_msgs, normalize_time=False
    )

    obj_topic = ros_utils.vicon_topic_name(vicon_object_name)
    obj_msgs = [msg for _, msg, _ in bag.read_messages(obj_topic)]
    obj_ts, obj_poses = ros_utils.parse_transform_stamped_msgs(
        obj_msgs, normalize_time=False
    )
    obj_poses = np.array(ros_utils.interpolate_list(ts, obj_ts, obj_poses))
    ts -= ts[0]

    # compute angle of tray from horizontal (which is equal to angle from the
    # z-axis)
    z = np.array([0, 0, 1])
    n = ts.shape[0]
    angles = np.zeros(n)
    for i in range(n):
        C_we = quat_to_rot(tray_poses[i, 3:])
        normal = C_we @ z
        angles[i] = np.arccos(z @ normal)

    # corresponding coefficients of friction
    mus = np.tan(angles)

    # compute offset of object from tray in tray's local frame
    r_locals = np.zeros((n, 3))
    for i in range(n):
        C_we = quat_to_rot(tray_poses[i, 3:])
        r_world = obj_poses[i, :3] - tray_poses[i, :3]
        r_locals[i, :] = C_we.T @ r_world

    # normalize by initial offset
    r_locals -= r_locals[0, :]

    # slip is the distance in the contact (x-y) plane
    slips = np.linalg.norm(r_locals[:, :2], axis=1)

    # find time and mu when object has slipped by 5mm
    slip_time = None
    for i in range(n):
        if ts[i] >= ignore_first_seconds and np.abs(slips[i]) >= slip_margin:
            break
    slip_time = ts[i]
    mu = mus[i]

    return FrictionData(
        ts=ts,
        angles=angles,
        mus=mus,
        slips=slips,
        mu=mu,
        slip_time=slip_time,
    )


