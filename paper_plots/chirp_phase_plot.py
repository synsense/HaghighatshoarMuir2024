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
from scipy.signal import hilbert, medfilt, lfilter
from pathlib import Path
import os
from tqdm import tqdm


def use_latex():
    matplotlib.use("pgf")
    matplotlib.rcParams.update(
        {
            "pgf.texsystem": "xelatex",
            "font.family": "serif",
            "text.usetex": True,
            "pgf.rcfonts": False,
        }
    )


use_latex()


def plot_phase_chirp():
    # find the directory for this file
    root = os.path.join(Path(__file__).resolve().parent, "phase_chirp_plot")

    if not os.path.exists(root):
        os.mkdir(root)

    filename = os.path.join(root, "phase_chirp.pdf")

    # chirp information
    duartion = 4
    period = 2
    fmin = 1_000
    fmax = 2_000
    fs = 100_000

    time_vec = np.arange(0, duartion, step=1 / fs)
    freq_inst = fmin + (fmax - fmin) * np.mod(time_vec, period) / period

    phase_inst = 2 * np.pi * np.cumsum(freq_inst) * 1 / fs

    sig_in = np.sin(phase_inst)

    sig_h = sig_in + 1j * hilbert(sig_in)
    phase_est = np.unwrap(np.angle(sig_h))

    win_size = 100
    filt_b = [1 / win_size]
    filt_a = [1, -(1 - 1 / win_size)]
    # freq_est = 1/(2*np.pi) * medfilt(np.diff(phase_est) / (1/fs), kernel_size=201)
    freq_est = lfilter(filt_b, filt_a, 1 / (2 * np.pi) * np.diff(phase_est) / (1 / fs))

    plt.figure()

    plt.subplot(211)
    plt.plot(time_vec[:-1], freq_est, linewidth=2, label=r"estimated $f_{est}(t)$")
    plt.plot(time_vec, freq_inst, linewidth=2, label=r"$f_{inst}(t)$")

    plt.legend()
    plt.title(
        f"chirp signal in frequency range [{fmin/1000}, {fmax/1000}] KHz, period: {period} sec"
    )
    plt.ylabel("instant frequency")
    plt.grid(True)

    plt.subplot(212)
    plt.plot(time_vec, phase_inst, linewidth=2, label=r"$\phi_{inst}(t)$")
    plt.plot(
        time_vec,
        phase_est,
        "-.",
        linewidth=2,
        label=r"estimated $\phi_{est}(t)$ using HT",
    )
    plt.ylabel("instant phase")
    plt.xlabel("time (s)")
    plt.legend()
    plt.grid(True)
    # plt.show()

    plt.savefig(filename)


def main():
    plot_phase_chirp()


if __name__ == "__main__":
    main()
