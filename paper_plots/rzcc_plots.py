# ----------------------------------------------------------------------------------------------------------------------
# This module produces plots for evaluating the Hilbert transform for harmonic signals
#
#
# (C) Saeid Haghighatshoar
# email: saeid.haghighatshoar@synsense.ai
#
#
# last update: 08.00.2023
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import hilbert, lfilter, freqz
from micloc.spike_encoder import ZeroCrossingSpikeEncoder

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
            "grid.linewidth": 0.5,
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


SAVE_PLOTS = True

if SAVE_PLOTS:
    use_latex()


def plot_RZCC():
    # find the directory for this file
    root = os.path.join(Path(__file__).resolve().parent, "RZCC_plots")

    if not os.path.exists(root):
        os.mkdir(root)

    # parameters of the system
    fs = 48_000
    freq_list = np.asarray([500])

    # signal duration
    sig_duration = 1 / np.min(freq_list)
    sig_len = int(fs * sig_duration)
    time_vec = np.linspace(0, sig_duration, sig_len)

    mm = 1 / 25.4

    plt.show()

    for freq in tqdm(freq_list):
        filename = os.path.join(root, f"rzcc_freq={freq}Hz.pdf")

        mm = 1 / 25.4

        sig_in = np.cos(2 * np.pi * freq * time_vec)
        # Make a spike encoder
        freq_range = [freq / 2, 2 * freq]
        zc_dist = int(fs / freq_range[-1])
        robust_width = zc_dist // 2

        print(robust_width)
        rzcc = ZeroCrossingSpikeEncoder(fs, robust_width)

        plt.figure(figsize=[60 * mm, 40 * mm])
        sig_cat = np.stack([sig_in, sig_in])
        spikes = rzcc.evolve(sig_cat.T)
        # spikes[spikes < 0] = 0

        plt.plot(time_vec * 1e3, sig_in, label="input signal")

        plt.plot(time_vec * 1e3, np.cumsum(sig_in) / 10 - 3.5, "C0--")

        plt.plot(0.5 + robust_width * np.array([-1, 1]) / fs * 1e3, [-1.5, -1.5], "k-")

        # plt.plot(spikes)
        plt.stem(time_vec * 1e3, spikes[:, 0] / 2, "C3-", markerfmt="", basefmt="none")

        plt.plot(time_vec[[0, -1]] * 1e3, [0, 0], "k:")
        # plt.legend()
        plt.grid(False)
        plt.axis("off")
        # plt.xlabel("Time (ms)")
        # plt.ylabel("Amplitude")
        # plt.title(
        #     f"STHT: f: {freq} Hz, {r'$f_s$'}: {int(fs/1000)} KHz, kernel-dur: {1000*win_duration:0.1f} ms"
        # )

        if SAVE_PLOTS:
            plt.savefig(filename, bbox_inches="tight", transparent=True)
        else:
            plt.draw()

        print(
            f"simulation was done for freq: {freq} and plot was saved in file:{filename}"
        )

    if SAVE_PLOTS:
        pass
    else:
        plt.show()


def main():
    plot_RZCC()


if __name__ == "__main__":
    main()
