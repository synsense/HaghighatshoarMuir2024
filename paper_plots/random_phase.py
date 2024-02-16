# ----------------------------------------------------------------------------------------------------------------------
# This module produces the phase plot for a chirp signal and shows that it can be used to track the instantenous frequency.
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 28.08.2023
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import hilbert, medfilt, lfilter, butter
from pathlib import Path
import os
from tqdm import tqdm


def use_latex():
    matplotlib.use("pdf")
    matplotlib.rcParams.update(
        {
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.5,
            "lines.linewidth": 1.0,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2,
            "ytick.major.size": 2,
            #     "pgf.texsystem": "xelatex",
            #     "font.family": "Helvetica",
            #     "text.usetex": True,
            #     "pgf.rcfonts": False,
        }
    )

    SMALL_SIZE = 6
    MEDIUM_SIZE = 8
    BIGGER_SIZE = 10

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=SMALL_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=SMALL_SIZE)  # fontsize of the figure title


use_latex()


def plot_phase_random():
    # find the directory for this file
    root = os.path.join(Path(__file__).resolve().parent, "phase_random_plot")

    if not os.path.exists(root):
        os.mkdir(root)

    filename = os.path.join(root, "phase_random_plot.pdf")

    # chirp information
    fmin = 1_000
    fmax = 3_000
    num_period = 20
    duration = num_period / fmin
    fs = 100 * fmax

    time_vec = np.arange(0, duration, step=1 / fs)
    num_sample = len(time_vec)
    # butterworth filter parameters
    N = 4
    Wn = [fmin, fmax]
    b, a = butter(N, Wn, btype="pass", analog=False, output="ba", fs=fs)

    mm = 1 / 25.4

    num_sim = 5
    plt.figure(figsize=(40 * mm, 40 * mm))

    for sim in range(num_sim):
        noise = np.random.randn(num_sample)
        sig_in = lfilter(b, a, noise)

        sig_h = sig_in + 1j * hilbert(sig_in)
        phase_est = np.unwrap(np.angle(sig_h))

        slope = (phase_est[-1] - phase_est[0]) / duration / (2 * np.pi)

        plt.plot(time_vec * 1e3, phase_est, label=f"signal {sim}: slope:{int(slope)}")

    plt.plot(
        time_vec[[0, -1]] * 1e3, [0, (fmin + fmax) / 2 * 2 * np.pi * duration], "k--"
    )

    # plt.title(f"Random noise with flat spectrum in {fmin/1000}–{fmax/1000} kHz")
    # plt.legend()
    plt.box(True)
    plt.grid(False)
    plt.xlabel("Time (ms)")
    plt.ylabel("$\phi$ (rad.)")
    # plt.show()

    plt.savefig(filename, bbox_inches="tight", transparent=True)


def main():
    plot_phase_random()


if __name__ == "__main__":
    main()
