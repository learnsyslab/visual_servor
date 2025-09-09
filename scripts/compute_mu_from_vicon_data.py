#!/usr/bin/env python3
"""Compute coefficient of friction from Vicon data.

This assumes that the z-axis of the tray Vicon model is actually aligned with
the tray normal, and that the world frame z-axis is gravity-aligned.
"""
import argparse

import matplotlib.pyplot as plt
import visual_servor as vs


SLIP_MARGIN = 0.005  # 5 mm slip is considered "slipping"

# ignore this first bit of data, which is helpful when the system is noisy
IGNORE_FIRST_SECONDS = 3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bagfile", help="Bag file to plot.")
    parser.add_argument(
        "--object", required=True, help="Name of object in Vicon system."
    )
    parser.add_argument("--tray", required=True, help="Name of tray in Vicon system.")
    args = parser.parse_args()

    data = vs.compute_friction_data(
        bag_path=args.bagfile,
        vicon_tray_name=args.tray,
        vicon_object_name=args.object,
        ignore_first_seconds=IGNORE_FIRST_SECONDS,
        slip_margin=SLIP_MARGIN,
    )

    print(f"mu = {data.mu}")

    plt.plot(data.ts, data.angles, label="angle")
    plt.plot(data.ts, data.mus, label="mu")
    plt.plot(data.ts, data.slips, label="slip")
    plt.axvline(data.slip_time, color="k")
    plt.axhline(data.mu, color="k")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
