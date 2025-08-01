import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pybullet as pyb
import pyb_utils
import rospkg
from xacrodoc import XacroDoc
from qpsolvers import solve_qp

import mobile_manipulation_central as mm

import IPython


TOOL_LINK_NAME = "ur10_arm_tool0"
ACTUATED_JOINT_NAMES = [
    "x_to_world_joint",
    "y_to_x_joint",
    "base_to_y_joint",
    "ur10_arm_shoulder_pan_joint",
    "ur10_arm_shoulder_lift_joint",
    "ur10_arm_elbow_joint",
    "ur10_arm_wrist_1_joint",
    "ur10_arm_wrist_2_joint",
    "ur10_arm_wrist_3_joint",
]
TIMESTEP = 0.01
DURATION = 10.0


def main():
    np.set_printoptions(precision=6, suppress=True)

    rospack = rospkg.RosPack()
    home_config_file = Path(rospack.get_path("visual_servor")) / "config/home.yaml"
    home = mm.load_home_position(name="sim", path=home_config_file)

    sim = mm.BulletSimulation(TIMESTEP)

    xacro_doc = XacroDoc.from_includes(
        [
            "$(find mobile_manipulation_central)/urdf/xacro/thing_pyb.urdf.xacro",
            "$(find visual_servor)/urdf/tray_pendulum.urdf.xacro",
        ]
    )
    with xacro_doc.temp_urdf_file_path() as urdf_path:
        robot_id = pyb.loadURDF(
            urdf_path,
            [0, 0, 0],
            useFixedBase=True,
        )

    robot = pyb_utils.Robot(
        robot_id,
        tool_link_name=TOOL_LINK_NAME,
        actuated_joint_names=ACTUATED_JOINT_NAMES,
    )
    robot.reset_joint_configuration(home)

    pendulum_joint_indices = [
        robot.get_joint_index(name)
        for name in ["pendulum_x_joint", "pendulum_y_joint", "pendulum_z_joint"]
    ]
    robot.set_joint_friction_forces(
        forces=[0.0, 0.0, 0.0], joint_indices=pendulum_joint_indices
    )
    tray_idx = robot.get_link_index("tray_pendulum")

    # QP params
    P = np.eye(9)
    qq = np.zeros(9)

    # no base velocity
    A1 = np.hstack((np.eye(3), np.zeros((3, 6))))
    b1 = np.zeros(3)

    lin_vel_des = np.zeros(3)
    tray_was_vertical = False

    r_ee_d, _ = robot.get_link_com_pose()
    r_tray_d, _ = robot.get_link_com_pose(link_idx=tray_idx)
    c_d = r_tray_d - r_ee_d

    ts = []
    ω_trays = []
    Δr_ees = []

    t = 0
    i = 0
    while t <= DURATION:
        q, v = robot.get_joint_states()
        J = robot.compute_link_jacobian(q)[:, :9]

        v_ee, _ = robot.get_link_com_velocity()
        v_tray, ω_tray = robot.get_link_com_velocity(link_idx=tray_idx)

        # detect when the tray is just about vertical
        r_ee, _ = robot.get_link_com_pose()
        r_tray, _ = robot.get_link_com_pose(link_idx=tray_idx)
        if r_tray[2] - r_ee[2] > 0.28:
            tray_was_vertical = True

        if t < 1.0:
            # get things started
            cmd_vel = np.zeros(9)
            cmd_vel[1] = t
            cmd_vel[0] = 0.5*t

            r_ee_d = r_ee
            r_tray_d = r_tray

        elif not tray_was_vertical:
            # pump energy into the tray

            c = r_tray - r_ee
            Δr_ee = r_ee_d - r_ee

            # stabilize tray
            u = 2*np.cross(ω_tray, c) - v_ee + Δr_ee
            # u = np.cross(ω_tray, c) - lin_vel_des + Δr - (c @ lin_vel_des) * c
            # u = -lin_vel_des + Δr

            # u = -lin_vel_des + Δr
            lin_vel_des += TIMESTEP * u

            # lin_vel_des = np.cross(ω_tray, r_tray - r_ee) + Δr

            # TODO limits on the actual EE velocity? I can just clamp it here
            A = np.vstack((A1, J))
            b = np.concatenate((b1, lin_vel_des, np.zeros(3)))
            cmd_vel = solve_qp(P=P, q=qq, A=A, b=b, solver="proxqp")
        else:
            cmd_vel = np.zeros(9)

        robot.command_velocity(cmd_vel)

        sim.step(t)
        i += 1
        t = i * TIMESTEP

        ts.append(t)
        Δr_ees.append(r_ee_d - r_ee)
        ω_trays.append(ω_tray)

        time.sleep(TIMESTEP)

    ts = np.array(ts)
    ω_trays = np.array(ω_trays)
    Δr_ees = np.array(Δr_ees)

    plt.figure()
    plt.title("EE position")
    plt.plot(ts, Δr_ees[:, 1], label="y")
    plt.xlabel("Time [s]")
    plt.ylabel("y [m]")
    plt.grid()

    plt.figure()
    plt.title("Tray angular velocity")
    plt.plot(ts, ω_trays[:, 0], label="ω_x")
    plt.xlabel("Time [s]")
    plt.ylabel("ω [rad/s]")
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()
