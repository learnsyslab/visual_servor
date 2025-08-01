import numpy as np
import matplotlib.pyplot as plt

import visual_servor as vs


def main():
    traj = vs.TrapezoidalTrajectory(a=0.5, t1=2, t2=2)

    ts = np.linspace(0, traj.duration, 1000)
    vs = np.zeros_like(ts)
    qs = np.zeros_like(ts)

    for i in range(ts.shape[0]):
        qs[i], vs[i] = traj.sample(ts[i])

    plt.figure()

    plt.plot(ts, qs, label="q")
    plt.plot(ts, vs, label="v")
    plt.grid()
    plt.legend()
    plt.xlabel("Time")

    plt.show()


if __name__ == "__main__":
    main()
