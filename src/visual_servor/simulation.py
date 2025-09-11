import numpy as np
import matplotlib.pyplot as plt


class SimulationData:
    """Data from a simulation run.

    Parameters
    ----------
    ts : array-like, shape (N,)
        Time steps.
    ss : array-like, shape (N,)
        Friction constraint slack variables.
    us : array-like, shape (N, 3)
        Global-frame input accelerations.
    φs : array-like, shape (N, 3)
        Tray orientation vectors.
    fcs : array-like, shape (N, 4, 3)
        Contact forces at each contact point.
    rs : array-like, shape (N, 3)
        Global-frame tray positions.
    rds : array-like, shape (N, 3)
        Global-frame desired tray positions.
    r_dots : array-like, shape (N, 3)
        Global-frame tray linear velocities.
    ωs : array-like, shape (N, 3)
        Global-frame tray angular velocities.
    """

    def __init__(
        self,
        t_interval,
        t_wait,
        t_stabilize,
        ts,
        ss,
        us,
        φs,
        fcs,
        rs,
        rds,
        r_dots,
        ωs,
        r_trays,
        r_tray_dots,
    ):
        self.ts = np.array(ts)
        self.ss = np.array(ss)
        self.us = np.array(us)
        self.φs = np.array(φs)
        self.fcs = np.array(fcs)
        self.rs = np.array(rs)
        self.rds = np.array(rds)
        self.r_dots = np.array(r_dots)
        self.ωs = np.array(ωs)
        self.r_trays = np.array(r_trays)
        self.r_tray_dots = np.array(r_tray_dots)

        # trajectory timing
        self.t_interval = t_interval
        self.t_wait = t_wait
        self.t_stabilize = t_stabilize
        self.stabilize_start_time = 3 * t_interval + t_wait

        # φs[φs < 0] += 2 * np.pi

        self.fts = np.linalg.norm(self.fcs[:, :, :2], axis=2)
        self.fns = self.fcs[:, :, 2]
        self.μs = self.fts / self.fns
        self.μ_max = np.max(self.μs)
        self.s_min = np.min(self.ss)

        # first time of friction constraint violation
        fail_idx = np.argmax(self.ss < 0)
        self.t_fail = self.ts[fail_idx] if fail_idx > 0 else None

    @classmethod
    def load(cls, filename):
        data = np.load(filename)
        return cls(
            t_interval=data["t_interval"].item(),
            t_wait=data["t_wait"].item(),
            t_stabilize=data["t_stabilize"].item(),
            ts=data["ts"],
            ss=data["ss"],
            us=data["us"],
            φs=data["φs"],
            fcs=data["fcs"],
            rs=data["rs"],
            rds=data["rds"],
            r_dots=data["r_dots"],
            ωs=data["ωs"],
            r_trays=data["r_trays"],
            r_tray_dots=data["r_tray_dots"],
        )

    def save(self, filename):
        np.savez(
            filename,
            t_interval=self.t_interval,
            t_wait=self.t_wait,
            t_stabilize=self.t_stabilize,
            ts=self.ts,
            ss=self.ss,
            us=self.us,
            φs=self.φs,
            fcs=self.fcs,
            rs=self.rs,
            rds=self.rds,
            r_dots=self.r_dots,
            ωs=self.ωs,
            r_trays=self.r_trays,
            r_tray_dots=self.r_tray_dots,
        )

    def plot(self):
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        plt.figure()
        plt.title("Input Acceleration")
        plt.plot(self.ts, self.us[:, 0], label="x")
        plt.plot(self.ts, self.us[:, 1], label="y")
        plt.plot(self.ts, self.us[:, 2], label="z")
        plt.grid()
        plt.xlabel("Time [s]")
        plt.ylabel("Acceleration [m/s^2]")
        plt.legend()

        plt.figure()
        plt.title("Position")
        plt.plot(self.ts, self.rs[:, 0], label="x")
        plt.plot(self.ts, self.rs[:, 1], label="y")
        plt.plot(self.ts, self.rs[:, 2], label="z")
        plt.plot(self.ts, self.rds[:, 0], linestyle="--", label="xd", color=colors[0])
        plt.plot(self.ts, self.rds[:, 1], linestyle="--", label="yd", color=colors[1])
        plt.plot(self.ts, self.rds[:, 2], linestyle="--", label="zd", color=colors[2])
        plt.grid()
        plt.xlabel("Time [s]")
        plt.ylabel("Position [m]")
        plt.legend()

        plt.figure()
        plt.title("Orientation")
        plt.plot(self.ts, self.φs[:, 0], label="x")
        plt.plot(self.ts, self.φs[:, 1], label="y")
        plt.plot(self.ts, self.φs[:, 2], label="z")
        plt.grid()
        plt.xlabel("Time [s]")
        plt.ylabel("Angle [rad]")
        plt.legend()

        plt.figure()
        plt.title("Linear velocity")
        plt.plot(self.ts, self.r_dots[:, 0], label="x")
        plt.plot(self.ts, self.r_dots[:, 1], label="y")
        plt.plot(self.ts, self.r_dots[:, 2], label="z")
        plt.grid()
        plt.xlabel("Time [s]")
        plt.ylabel("Linear velocity [m/s]")
        plt.legend()

        plt.figure()
        plt.title("Angular velocity")
        plt.plot(self.ts, self.ωs[:, 0], label="x")
        plt.plot(self.ts, self.ωs[:, 1], label="y")
        plt.plot(self.ts, self.ωs[:, 2], label="z")
        plt.grid()
        plt.xlabel("Time [s]")
        plt.ylabel("Angular velocity [rad/s]")
        plt.legend()

        plt.figure()
        plt.title("Contact forces")
        plt.plot(self.ts, self.fcs[:, 0, 0], label="x")
        plt.plot(self.ts, self.fcs[:, 0, 1], label="y")
        plt.plot(self.ts, self.fcs[:, 0, 2], label="z")
        plt.grid()
        plt.xlabel("Time [s]")
        plt.ylabel("Contact force [N]")
        plt.legend()

        plt.show()
